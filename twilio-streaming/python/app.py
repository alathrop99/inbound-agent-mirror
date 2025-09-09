import base64
import json
import os
import sys
import time
import wave
import audioop
from pathlib import Path
import numpy as np
from flask import Flask, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse
# Optional DSP for resampling
try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
# LLM + RAG + ASR
# Ensure project root is on sys.path so we can import rag.py, etc.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from pathlib import Path
# Ensure project root is on sys.path so we can import rag.py, etc.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
try:
    import rag as rag_mod  # may fail if faiss/numpy mismatch
    _RAG_OK = True
except Exception as e:
    print("[RAG] Import failed; using fallback retriever:", e)
    rag_mod = None
    _RAG_OK = False
from faster_whisper import WhisperModel
import ollama
# Optional Kokoro TTS backend (preferred)
try:
    from kokoro import KPipeline as KokoroPipeline
    _HAVE_KOKORO = True
except Exception:
    _HAVE_KOKORO = False
try:
    import torch as _torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False
app = Flask(__name__)
sock = Sock(app)
# ------------------
# Config
# ------------------
SYSTEM_PROMPT = (
    "You are Alley, a friendly, confident AI assistant for Curve Dental. "
    "Speak like a knowledgeable rep — natural, direct, and concise. "
    "Only respond to what was asked. Use plain, human language. "
    "Prioritize clarity over detail. One or two sentences is ideal — three max. "
    "Don't overload with extra info or stats unless it directly answers the question. "
    "You'll be given trusted background info. Use that only. If unsure, say so briefly. "
    "Never cite sources or say 'according to the knowledge base.'"
)
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct")
# Point RAG at the repo's knowledge folder explicitly (works regardless of CWD)
KNOWLEDGE_DIR = str(ROOT_DIR / "knowledge")
if _RAG_OK:
    rag_mod.DATA_DIR = KNOWLEDGE_DIR
    rag_mod.ALIAS_FILE = os.path.join(rag_mod.DATA_DIR, "aliases.json")
    rag_mod.INDEX_FILE = os.path.join(rag_mod.DATA_DIR, "kb.index")
    rag_mod.META_FILE  = os.path.join(rag_mod.DATA_DIR, "kb_meta.npy")
    RAG = rag_mod.SimpleRAG()
else:
    # Minimal file-based fallback retriever (no FAISS)
    import re
    from glob import glob
    try:
        from rapidfuzz import process, fuzz
        _HAVE_RF = True
    except Exception:
        _HAVE_RF = False
    class FallbackRAG:
        def __init__(self, data_dir: str):
            self.entries = []  # list of (text, source)
            exts = ("*.txt", "*.md")
            for pat in exts:
                for p in glob(os.path.join(data_dir, "**", pat), recursive=True):
                    try:
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            txt = f.read()
                    except Exception:
                        continue
                    # Split on sentences/lines
                    parts = re.split(r"(?<=[\.!?])\s+|\n+", txt)
                    for s in parts:
                        s2 = s.strip()
                        if len(s2) >= 6:
                            self.entries.append((s2, os.path.relpath(p, data_dir)))
        def _normalize(self, q: str) -> str:
            ql = q.lower()
            # simple alias corrections to help ASR confusions
            repl = {
                "leave": "weave",
                "weeve": "weave",
                "weve": "weave",
                "dent ricks": "dentrix",
                "den tricks": "dentrix",
                "eagle soft": "eaglesoft",
                "opendental": "open dental",
            }
            for a, b in repl.items():
                ql = ql.replace(a, b)
            return ql
        def retrieve(self, query: str, k: int = 5):
            qn = self._normalize(query)
            toks = [t for t in re.findall(r"[a-z0-9]+", qn) if len(t) > 2]
            hits = []
            if _HAVE_RF:
                # Use rapidfuzz over the text list for better matching
                candidates = [t for (t, _s) in self.entries]
                rf = process.extract(qn, candidates, scorer=fuzz.token_set_ratio, limit=max(k*4, 10))
                # map back to entries indices
                for cand, score, idx in rf:
                    t, s = self.entries[idx]
                    hits.append((score, t, s))
            else:
                for (t, s) in self.entries:
                    tl = t.lower()
                    score = sum(1 for tok in toks if tok in tl)
                    if score:
                        hits.append((score, t, s))
            hits.sort(reverse=True, key=lambda x: x[0])
            out = []
            for _score, t, s in hits[:k]:
                out.append((t, s))
            return out
    RAG = FallbackRAG(KNOWLEDGE_DIR)
# Whisper for 16kHz float32 mono input
try:
    ASR = WhisperModel("base", device="cuda", compute_type="float16")
except Exception:
    ASR = WhisperModel("base", device="cpu", compute_type="int8")
# ------------------
# Helpers: Twilio media <-> PCM
# ------------------
MU = 255.0
TTS_RATE = int(os.environ.get("TTS_RATE", "200"))  # pyttsx3 rate (fallback)
FRAME_MS = int(os.environ.get("TTS_FRAME_MS", "20"))  # Twilio expects 20ms
TTS_BACKEND = os.environ.get("TTS_BACKEND", "KOKORO").upper()  # KOKORO | PYTTSX3
KOKORO_LANG_CODE = os.environ.get("KOKORO_LANG_CODE", "a")  # 'a' = American English
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
KOKORO_DEVICE = os.environ.get("KOKORO_DEVICE") or ("cuda" if (_HAVE_TORCH and _torch.cuda.is_available()) else "cpu")
_kokoro_synth = None
def mulaw_decode(u8: np.ndarray) -> np.ndarray:
    """Decode µ-law bytes to float32 PCM -1..1 using audioop for correctness."""
    if u8 is None or len(u8) == 0:
        return np.zeros(0, dtype=np.float32)
    # audioop.ulaw2lin expects a bytes-like object
    lin16 = audioop.ulaw2lin(u8.tobytes(), 2)  # 16-bit linear PCM
    pcm16 = np.frombuffer(lin16, dtype=np.int16)
    x = (pcm16.astype(np.float32)) / 32768.0
    return x
def mulaw_encode(x: np.ndarray) -> np.ndarray:
    """Encode float32 PCM -1..1 to µ-law using audioop for correct G.711.
    Returns uint8 array of mu-law bytes.
    """
    # Normalize and convert to int16 linear PCM
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    # Avoid clipping; normalize to 0.9 if too loud
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > 1e-6:
        scale = min(0.9 / m, 1.0)
        x = x * scale
    x = np.clip(x, -1.0, 1.0)
    pcm16 = (x * 32767.0).astype(np.int16)
    # audioop.lin2ulaw expects bytes of 16-bit linear PCM
    ulaw_bytes = audioop.lin2ulaw(pcm16.tobytes(), 2)
    return np.frombuffer(ulaw_bytes, dtype=np.uint8)
def resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == 16000:
        return x
    if _HAVE_SCIPY:
        return resample_poly(x, 16000, sr_in)
    ratio = 16000 / float(sr_in)
    idx = (np.arange(int(len(x) * ratio)) / ratio).astype(np.int64)
    idx = np.clip(idx, 0, len(x) - 1)
    return x[idx]
def resample_to_8k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == 8000:
        return x
    if _HAVE_SCIPY:
        return resample_poly(x, 8000, sr_in)
    ratio = 8000 / float(sr_in)
    idx = (np.arange(int(len(x) * ratio)) / ratio).astype(np.int64)
    idx = np.clip(idx, 0, len(x) - 1)
    return x[idx]
def build_context_block(query: str, k: int = 5) -> str:
    try:
        hits = RAG.retrieve(query, k=k)
    except Exception:
        hits = []
    if not hits:
        return ""
    lines = [f"- {t}" for (t, _src) in hits]
    return (
        "Use these verified facts. Prefer them over prior assumptions. "
        "Do not include citations or bracket codes; answer naturally.\n"
        + "\n".join(lines)
    )
def llm_reply(user_text: str) -> str:
    # Retrieve facts; if none, avoid hallucinations by responding conservatively
    ctx = build_context_block(user_text)
    if not ctx:
        return (
            "I'm not totally sure on that. I can check and follow up, "
            "or we can look at features, integrations, or training details."
        )
    msgs = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "system", "content": ctx},
    # Priming examples to improve conversational tone
    {"role": "user", "content": "Do you integrate with Weave?"},
    {"role": "assistant", "content": "Not at the moment, but that’s something we may support in the future. I’d be happy to check if there’s a workaround."},
    {"role": "user", "content": "How long does training usually take?"},
    {"role": "assistant", "content": "Most offices are up and running in a few days. Training is flexible — some finish in one or two sessions, others spread it out."},
    {"role": "user", "content": "Why switch to Curve from Dentrix?"},
    {"role": "assistant", "content": "Dentrix is server-based and can be clunky. Curve is cloud-native, includes more built-in tools, and doesn’t need IT help to run."},
    # Actual user question
    {"role": "user", "content": user_text},
    ]
    out = []
    for event in ollama.chat(model=MODEL_NAME, messages=msgs, stream=True, options={
    "num_ctx": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_predict": 100  # ~2 sentences max
}):
        if "message" in event and "content" in event["message"]:
            out.append(event["message"]["content"])
    reply = "".join(out).strip()
    # sanitize any stray [F#]
    import re as _re
    reply = _re.sub(r"\s*\[F\d+\]\s*", " ", reply)
    reply = _re.sub(r"\s{2,}", " ", reply)
    return reply
def _ensure_kokoro_loaded():
    global _kokoro_synth
    if _kokoro_synth is None:
        if not _HAVE_KOKORO:
            raise RuntimeError("Kokoro TTS not available; install 'kokoro' package or set TTS_BACKEND=PYTTSX3")
        # Replace with pipeline instance compatible with kokoro_test.py usage
        _kokoro_synth = KokoroPipeline(lang_code=KOKORO_LANG_CODE, device=KOKORO_DEVICE)
def tts_to_pcm(text: str, sr_out: int = 8000) -> np.ndarray:
    """Synthesize text to PCM float32 mono at sr_out (default 8k)."""
    import tempfile
    # Prefer Kokoro if requested and available
    if TTS_BACKEND == "KOKORO" and _HAVE_KOKORO:
        _ensure_kokoro_loaded()
        # Generate audio array directly via pipeline
        gen = _kokoro_synth(text, voice=KOKORO_VOICE)
        try:
            _gtxt, _ph, audio = next(gen)
        except StopIteration:
            audio = np.zeros(0, dtype=np.float32)
        sr_in = getattr(_kokoro_synth, "sample_rate", 24000)
        x = np.asarray(audio, dtype=np.float32)
        # Resample to 8k
        x8k = resample_to_8k(x, sr_in)
        return x8k.astype(np.float32)
    else:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
    # Read wav (pyttsx3 path)
    with wave.open(tmp_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr_in = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    # Ensure 16-bit linear PCM for uniform processing
    if sampwidth != 2:
        try:
            raw = audioop.lin2lin(raw, sampwidth, 2)
            sampwidth = 2
        except Exception:
            # Fallback assume 16-bit as-is
            pass
    # Convert to float32 mono
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    if n_channels and n_channels > 1:
        try:
            pcm16 = pcm16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
        except Exception:
            pass
    x = (pcm16.astype(np.float32)) / 32768.0
    # Light normalization to avoid clipping and ensure audibility
    mx = float(np.max(np.abs(x))) if x.size else 0.0
    if mx > 1e-6:
        x = x * (0.9 / mx)
    # Resample to 8k
    x8k = resample_to_8k(x, sr_in)
    return x8k.astype(np.float32)
def stream_tts(ws, stream_sid: str, text: str):
    """Send synthesized speech back to caller as 8k µ-law frames paced to real-time.
    Uses time-based scheduling to avoid choppy/slow playback.
    """
    pcm = tts_to_pcm(text, sr_out=8000)
    u = mulaw_encode(pcm)
    samples_per_frame = int(8000 * (FRAME_MS / 1000.0))  # default 160 samples at 20ms
    # Pre-slice frames and base64 encode upfront to reduce per-iteration work
    payloads = []
    for i in range(0, len(u), samples_per_frame):
        chunk = u[i:i + samples_per_frame]
        if len(chunk) == 0:
            continue
        payloads.append(base64.b64encode(chunk.tobytes()).decode("ascii"))
    start = time.perf_counter()
    frame_sec = FRAME_MS / 1000.0
    for idx, payload in enumerate(payloads):
        target = start + idx * frame_sec
        now = time.perf_counter()
        delay = target - now
        # If we are ahead of schedule, wait; if behind, send immediately
        if delay > 0:
            # Slightly under-sleep to account for send overhead
            time.sleep(max(0.0, delay * 0.9))
        # Send frame
        ws.send(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload, "track": "outbound"}
        }))
    try:
        ws.send(json.dumps({
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": "tts-complete"}
        }))
    except Exception:
        pass
# ------------------
# Routes
# ------------------
@app.route("/twilio-voice", methods=["POST"])
def twilio_voice():
    response = VoiceResponse()
    stream_url = os.environ.get("TWILIO_STREAM_WSS", "wss://b27ab4aa24de.ngrok-free.app/media")
    response.connect().stream(url=stream_url)
    return Response(str(response), mimetype="text/xml")
@sock.route('/media')
def media_stream(ws):
    print("[Twilio] WebSocket connection established")
    stream_sid = None
    # Simple VAD over 20ms frames @8k
    chunk_ms = 20
    silence_ms_needed = 500
    silence_thresh = 0.006
    silent_needed = silence_ms_needed // chunk_ms
    silent_streak = 0
    capturing = False
    buf = []
    media_count = 0
    while True:
        msg = ws.receive()
        if msg is None:
            break
        try:
            data = json.loads(msg)
        except Exception:
            continue
        event = data.get("event")
        if event == "start":
            stream_sid = data.get("start", {}).get("streamSid")
            print(f"[Twilio] start streamSid={stream_sid}")
            # Greet caller immediately so we verify outbound audio path
            try:
                stream_tts(ws, stream_sid, "Hi, thank you for calling Curve Dental, this is Alley, an A I assistant. How can I help today?")
            except Exception as e:
                print(f"[Twilio] TTS greet error: {e}")
            continue
        if event == "stop":
            print("[Twilio] stop")
            break
        if event != "media":
            continue
        media = data.get("media", {})
        payload = media.get("payload")
        if not payload:
            continue
        try:
            b = base64.b64decode(payload)
            u8 = np.frombuffer(b, dtype=np.uint8)
            x = mulaw_decode(u8)
        except Exception:
            continue
        media_count += 1
        if media_count % 50 == 0:
            print(f"[Twilio] media frames: {media_count}")
        if len(x) == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(x))))
        is_silent = rms < silence_thresh
        if not capturing and not is_silent:
            capturing = True
            buf = []
            silent_streak = 0
        if capturing:
            buf.append(x)
            silent_streak = (silent_streak + 1) if is_silent else 0
            if silent_streak >= silent_needed and len(buf) > 0:
                audio8 = np.concatenate(buf, axis=0)
                capturing = False
                silent_streak = 0
                buf = []
                audio16 = resample_to_16k(audio8, 8000)
                try:
                    segments, info = ASR.transcribe(audio16, beam_size=1)
                    text = "".join(seg.text for seg in segments).strip()
                except Exception:
                    text = ""
                if not text:
                    continue
                print(f"[ASR] {text}")
                try:
                    reply = llm_reply(text)
                except Exception as e:
                    print(f"[LLM] error: {e}")
                    reply = "Sorry, I had a problem processing that. Could you repeat?"
                print(f"[LLM] {reply}")
                if stream_sid:
                    stream_tts(ws, stream_sid, reply)
    print("[Twilio] WebSocket connection closed")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
