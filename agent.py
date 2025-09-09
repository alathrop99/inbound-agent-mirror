import os
import time
import numpy as np
import pyttsx3
import threading
from queue import Queue
try:
    import pythoncom  # ensures COM init for SAPI on Windows threads
    _HAVE_PYWIN32 = True
except Exception:
    _HAVE_PYWIN32 = False
try:
    import win32com.client as win32
    _HAVE_WIN32COM = True
except Exception:
    _HAVE_WIN32COM = False
try:
    import torch as _torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False
import sounddevice as sd
from faster_whisper import WhisperModel
import ollama
from collections import deque
from rag import SimpleRAG
from time import perf_counter
import torch

# --- Sentence streaming helper ---
_SENT_END = ('.', '!', '?')

# --- streaming sentence flusher ---
def flush_sentences(accum):
    """Flush complete sentences from accumulated LLM stream."""
    from re import split
    buf = accum.get("buf", "")
    sents = []

    # Split on sentence-ending punctuation
    parts = split(r"(?<=[.!?])\s+", buf)

    if len(parts) > 1:
        sents, tail = parts[:-1], parts[-1]
    else:
        return []

    accum["buf"] = tail
    return sents

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "mistral:7b-instruct"   # pulled in Milestone A
print(f"[Config] Using model: {MODEL_NAME}")
ASR_MODEL  = "base"                  # try "small" or "medium" later if you want
SAMPLE_RATE = 16000                  # 16 kHz mono keeps things simple

SYSTEM_PROMPT = (
    "You are Alley, a friendly, confident AI assistant for Curve Dental. "
    "Speak like a knowledgeable rep — natural, direct, and concise. "
    "Only respond to what was asked. Use plain, human language. "
    "Prioritize clarity over detail. One or two sentences is ideal — three max. "
    "Don't overload with extra info or stats unless it directly answers the question. "
    "You'll be given trusted background info. Use that only. If unsure, say so briefly. "
    "Never cite sources or say 'according to the knowledge base.'"
)

# ------- VAD tuning -------
VAD_CHUNK_MS = 200          # analyze in 200 ms chunks
VAD_SILENCE_MS = 1100        # need ~0.9s of continuous silence to stop
VAD_SILENCE_THRESH = 0.018  # RMS below this counts as "silent" (0..1)
VAD_WARMUP_MS = 400         # ignore silence for the first 0.4s
VAD_MIN_CAPTURE_MS = 1200   # don't stop before we capture at least 1.2s
VAD_PREROLL_MS = 300        # keep extra 0.3s before the first voice (anti-clipping)

# Keep the last few turns for context
HISTORY = deque(maxlen=8)  # (user, assistant) pairs

RAG = SimpleRAG(facts_file="knowledge/facts.txt")  # loads facts + builds index

# ---------------------------
# TTS init with background worker (Kokoro preferred; SAPI/pyttsx3 fallback)
# ---------------------------
_TTS_Q: Queue[str] = Queue()
_tts_thread = None
DEBUG_TTS = False
TTS_ENGINE = os.environ.get("TTS_ENGINE", "KOKORO").upper()  # KOKORO | SAPI | PYTTSX3
try:
    from kokoro import KPipeline as KokoroPipeline
    _HAVE_KOKORO = True
except Exception:
    _HAVE_KOKORO = False
KOKORO_LANG_CODE = os.environ.get("KOKORO_LANG_CODE", "a")  # 'a' = American English
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
KOKORO_DEVICE = os.environ.get("KOKORO_DEVICE") or ("cuda" if (_HAVE_TORCH and _torch.cuda.is_available()) else "cpu")
_kokoro_pipeline = None

def _tts_worker():
    try:
        use_kokoro = (TTS_ENGINE == "KOKORO") and _HAVE_KOKORO
        sapi_voice = None
        tts_engine = None
        if not use_kokoro:
            if _HAVE_PYWIN32:
                try:
                    pythoncom.CoInitialize()
                except Exception:
                    pass
        # Choose engine
        use_sapi = (not use_kokoro) and (TTS_ENGINE == "SAPI") and _HAVE_WIN32COM
        if use_sapi:
            try:
                sapi_voice = win32.Dispatch("SAPI.SpVoice")
                # Optional tuning
                try:
                    sapi_voice.Rate = 0  # -10..+10
                    sapi_voice.Volume = 100
                except Exception:
                    pass
            except Exception as e:
                print(f"[TTS] SAPI init failed: {e}; falling back to pyttsx3")
                use_sapi = False
        if use_kokoro:
            global _kokoro_pipeline
            if _kokoro_pipeline is None:
                _kokoro_pipeline = KokoroPipeline(lang_code=KOKORO_LANG_CODE, device=KOKORO_DEVICE)
        elif not use_sapi:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty("rate", 185)
            tts_engine.setProperty("volume", 1.0)
            # tts_engine.setProperty("voice", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0")
        while True:
            text = _TTS_Q.get()
            if text is None:
                _TTS_Q.task_done()
                break
            try:
                if DEBUG_TTS:
                    print(f"[TTS] speak start: {text}")
                if use_kokoro and _kokoro_pipeline is not None:
                    import numpy as _np
                    try:
                        gen = _kokoro_pipeline(text, voice=KOKORO_VOICE)
                        _gt, _ph, audio = next(gen)
                        sr = getattr(_kokoro_pipeline, "sample_rate", 24000)
                        x = _np.asarray(audio, dtype=_np.float32)
                        sd.play(x, sr)
                        sd.wait()
                    except StopIteration:
                        pass
                elif use_sapi and sapi_voice is not None:
                    sapi_voice.Speak(text)
                else:
                    tts_engine.say(text)
                    tts_engine.runAndWait()
                if DEBUG_TTS:
                    print("[TTS] speak done")
            except Exception as ex:
                print(f"[TTS] Error: {ex}")
                # Try to recover by reinitializing pyttsx3 engine
                try:
                    if not use_sapi:
                        tts_engine = pyttsx3.init()
                        tts_engine.setProperty("rate", 185)
                        tts_engine.setProperty("volume", 1.0)
                except Exception:
                    pass
            finally:
                _TTS_Q.task_done()
    except Exception as ex:
        print(f"[TTS] Init error: {ex}")
    finally:
        if (TTS_ENGINE != "KOKORO") and _HAVE_PYWIN32:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

def _ensure_tts_thread():
    global _tts_thread
    if _tts_thread is None or not _tts_thread.is_alive():
        _tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        _tts_thread.start()

def speak(text: str):
    # Enqueue speech to ensure all sentences are spoken sequentially.
    if text and text.strip():
        _ensure_tts_thread()
        _TTS_Q.put(text.strip())

def shutdown_tts():
    try:
        _TTS_Q.put(None)
        _TTS_Q.join()
    except Exception:
        pass

# ---------------------------
# ASR (faster-whisper) init
# ---------------------------
# device="auto" lets it use your GPU if possible, otherwise CPU.
# compute_type float16 on GPU; int8 on CPU for memory.
asr = WhisperModel(ASR_MODEL, device="cuda", compute_type="float16")

def record_until_silence(max_seconds: float = 10.0) -> np.ndarray:
    """
    Record mic until a stretch of silence is detected, with warmup + min capture.
    Returns mono float32 PCM at SAMPLE_RATE. Empty array if nothing captured.
    """
    print(f"\n[Mic] Speak now… (auto-stops on ~{VAD_SILENCE_MS} ms silence, max {max_seconds:.1f}s)")
    frames = []
    chunk_samples = int(SAMPLE_RATE * (VAD_CHUNK_MS / 1000.0))
    silent_needed = int(VAD_SILENCE_MS / VAD_CHUNK_MS)
    warmup_chunks = int(VAD_WARMUP_MS / VAD_CHUNK_MS)
    mincap_chunks = int(VAD_MIN_CAPTURE_MS / VAD_CHUNK_MS)
    preroll_chunks = int(VAD_PREROLL_MS / VAD_CHUNK_MS)

    silent_streak = 0
    voiced_seen = False
    first_voice_idx = None
    start_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        i = 0
        while True:
            chunk, _ = stream.read(chunk_samples)
            frames.append(chunk.copy())

            rms = float(np.sqrt(np.mean(np.square(chunk))))
            is_silent = (rms < VAD_SILENCE_THRESH) if i >= warmup_chunks else False

            # detect first voice after warmup
            if not voiced_seen and not is_silent and i >= warmup_chunks:
                voiced_seen = True
                first_voice_idx = max(0, i - preroll_chunks)  # keep a little pre-roll

            silent_streak = (silent_streak + 1) if is_silent else 0

            elapsed = time.time() - start_time
            i += 1

            # stop if enough silence AND we've recorded long enough AND we actually heard voice
            if (silent_streak >= silent_needed
                and i >= mincap_chunks
                and voiced_seen):
                print("[Mic] Detected silence — stopping.")
                break

            if elapsed >= max_seconds:
                print("[Mic] Max time reached — stopping.")
                break

    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).squeeze()

    # trim head if we never detected voice; otherwise keep from the preroll mark
    if voiced_seen and first_voice_idx is not None:
        start_sample = first_voice_idx * chunk_samples
        audio = audio[start_sample:]

    return audio

def transcribe(audio: np.ndarray) -> str:
    """Transcribe a float32 PCM array with faster-whisper."""
    # faster-whisper expects float32 -1..1; sounddevice already gives that.
    segments, info = asr.transcribe(audio, beam_size=1)
    text = "".join([seg.text for seg in segments]).strip()
    print(f"[ASR] {text}")
    return text

def build_context_block(query: str, k: int = 3):
    """
    Retrieve top-k facts and format as a block with simple citations.
    Returns (context_text, debug_list) where debug_list holds (text, source).
    """
    try:
        hits = RAG.retrieve(query, k=k)
    except Exception:
        hits = []
    if not hits:
        return "", []
    lines = []
    for i, (text, _src) in enumerate(hits, 1):
        # Present facts plainly without [F#] labels to discourage citation in replies
        lines.append(f"- {text}")
    ctx = (
        "Use these verified facts. Prefer them over prior assumptions. "
        "In your reply, do not include citations or bracket codes; answer naturally.\n"
        + "\n".join(lines)
    )
    return ctx, hits

def generate_llm_reply(user_text: str) -> str:
    t0 = perf_counter()
    context_block, hits = build_context_block(user_text, k=5)
    t_rag_end = perf_counter()

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_block:
        msgs.append({
            "role": "system",
            "content": 
                context_block
        })
    # Priming examples to improve conversational tone
    msgs += [
        {"role": "user", "content": "Do you integrate with Weave?"},
        {"role": "assistant", "content": "Not at the moment, but that’s something we may support in the future. I’d be happy to check if there’s a workaround."},

        {"role": "user", "content": "How long does training usually take?"},
        {"role": "assistant", "content": "Most offices are up and running in a few days. Training is flexible — some finish in one or two sessions, others spread it out."},

        {"role": "user", "content": "Why switch to Curve from Dentrix?"},
        {"role": "assistant", "content": "Dentrix is server-based and can be clunky. Curve is cloud-native, includes more built-in tools, and doesn’t need IT help to run."},
]

    for u, a in list(HISTORY):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})

    msgs.append({"role": "user", "content": user_text})

    first_token_time = None
    accum = {"buf": ""}
    full_reply = ""

    for event in ollama.chat(
        model=MODEL_NAME,
        messages=msgs,
        stream=True,
        options={
        "num_ctx": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_predict": 100  # ~2 sentences max
    }
    ):
        if "message" in event and "content" in event["message"]:
            if first_token_time is None:
                first_token_time = perf_counter()

            piece = event["message"]["content"]
            accum["buf"] += piece
            full_reply += piece

            # While streaming, flush full sentences
            sents = flush_sentences(accum)
            for s in sents:
                # Strip any bracket citations like [F1] just in case
                import re as _re
                s_clean = _re.sub(r"\s*\[F\d+\]\s*", " ", s).strip()
                s_clean = _re.sub(r"\s{2,}", " ", s_clean)
                if len(s_clean) > 6:
                    print(f"[SPEAKING] {s_clean}")
                    time.sleep(0.05)
                    speak(s_clean)  # queued; will play after prior items

    # 🔥 Final flush for any remaining partial
    leftover = accum.get("buf", "").strip()
    if leftover:
        import re as _re
        s_clean = _re.sub(r"\s*\[F\d+\]\s*", " ", leftover).strip()
        s_clean = _re.sub(r"\s{2,}", " ", s_clean)
        print(f"[SPEAKING] {s_clean}")
        time.sleep(0.05)
        speak(s_clean)  # queued

    # Append any leftover partial sentence
    import re as _re
    reply = _re.sub(r"\s*\[F\d+\]\s*", " ", full_reply).strip()
    reply = _re.sub(r"\s{2,}", " ", reply)
    t_llm_done = perf_counter()

    HISTORY.append((user_text, reply))
    print(f"[LLM] {reply}")

    first_delta = (first_token_time - t_rag_end) * 1000 if first_token_time else 0
    print(
        f"[TIMING] RAG: {(t_rag_end - t0)*1000:.0f} ms | "
        f"LLM first token: {first_delta:.0f} ms | "
        f"LLM total: {(t_llm_done - t_rag_end)*1000:.0f} ms"
    )

    return reply

def main():
    print("\n=== Local Voice Agent (MVP, multi-turn) ===")
    print("Press ENTER, speak, then pause—recording auto-stops on silence.")
    print("Type 'q' + ENTER to quit.\n")

    while True:
        cmd = input("Ready. Press ENTER to record, type 'text: ...' to type instead, 'reset' to clear context, or 'q' to quit: ").strip()

        if cmd.lower() == "q":
            print("Bye!")
            try:
                shutdown_tts()
            except Exception:
                pass
            break

        if cmd.lower() == "reset":
            HISTORY.clear()
            print("[Ctx] Cleared.")
            continue

        if cmd.lower().startswith("text:"):
            user_text = cmd[5:].strip()
            if not user_text:
                print("Nothing to send; try again.")
                continue
            reply = generate_llm_reply(user_text)
            continue

        if cmd.lower().startswith("facts:"):
            q = cmd[6:].strip() or "test"
            print("[Facts]", build_context_block(q, k=3) or "(no facts)")
            continue

        if cmd.lower() == "reload":
            try:
                RAG.rebuild()
                print("[RAG] Rebuilt index from knowledge/.")
            except Exception as e:
                print("[RAG] Rebuild failed:", e)
            continue

        if cmd.lower().startswith("why"):
            # shows which facts the last question would have pulled; or demo with a query
            q = cmd[3:].strip() or "test"
            ctx, hits = build_context_block(q, k=3)
            if not hits:
                print("[Why] (no matching facts)")
            else:
                print("[Why]\n" + ctx)
            continue

        # otherwise, go to mic
        audio = record_until_silence(max_seconds=10.0)
        if audio is None or len(audio) == 0:
            print("No audio captured; try again.")
            continue

        try:
            t_asr0 = perf_counter()
            user_text = transcribe(audio)
            t_asr1 = perf_counter()
            print(f"[TIMING] ASR: {(t_asr1 - t_asr0)*1000:.0f} ms")
            if not user_text:
                print("Heard nothing intelligible; try again.")
                continue

            reply = generate_llm_reply(user_text)

        except Exception as e:
            print(f"Error: {e}")
            speak("Sorry, something went wrong.")

if __name__ == "__main__":
    main()
