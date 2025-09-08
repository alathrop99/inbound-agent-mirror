import os
import re
import json
import shutil
import argparse
from pathlib import Path
from collections import OrderedDict

try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass

import requests


# ------------
# Config
# ------------
DATA_DIR = Path("data")
DEFAULT_INPUT_FOLDER = DATA_DIR / "calls_in"
DEFAULT_OUTPUT_FOLDER = DATA_DIR / "calls_out"
DEFAULT_ARCHIVE_FOLDER = DATA_DIR / "archived_txt"

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# ------------
# Transcript parsing
# ------------
HEADER_RE = re.compile(r"^(\d{1,2}:\d{2})\s*\|\s*(.+)$")


def parse_transcript(path: Path):
    """
    Parse transcripts with blocks like:
      0:00 | Ray
      Bradford dental, Ray speaking.
    Accumulate text until the next header line.
    Returns list of {speaker, timestamp, start_time, start_sec, text}.
    """
    convo = []
    current = None

    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.replace("\r", "").rstrip("\n")
            cleaned = line.lstrip("\ufeff").strip()

            m = HEADER_RE.match(cleaned)
            if m:
                # flush previous
                if current:
                    text = " ".join(t.strip() for t in current["text_lines"] if t.strip()).strip()
                    convo.append({
                        "speaker": current["speaker"],
                        "timestamp": current["timestamp"],
                        "start_time": current["timestamp"],
                        "start_sec": current["start_sec"],
                        "text": text,
                    })
                ts, speaker = m.groups()
                try:
                    mm, ss = ts.split(":"); start_sec = int(mm) * 60 + int(ss)
                except Exception:
                    start_sec = 0
                current = {
                    "timestamp": ts.strip(),
                    "speaker": speaker.strip(),
                    "start_sec": float(start_sec),
                    "text_lines": [],
                }
                continue

            if current is not None:
                if cleaned == "":
                    continue
                current["text_lines"].append(cleaned)

    if current:
        text = " ".join(t.strip() for t in current["text_lines"] if t.strip()).strip()
        convo.append({
            "speaker": current["speaker"],
            "timestamp": current["timestamp"],
            "start_time": current["timestamp"],
            "start_sec": current["start_sec"],
            "text": text,
        })

    return convo


def normalize_end_times(utterances, epsilon=0.05):
    for i, u in enumerate(utterances):
        if i < len(utterances) - 1:
            next_start = utterances[i + 1].get("start_sec", u.get("start_sec", 0) + 3.0)
            u["end_sec"] = max(u.get("start_sec", 0.0), next_start - epsilon)
        else:
            u["end_sec"] = u.get("start_sec", 0.0) + 3.0
    return utterances


# ------------
# Heuristic fallback
# ------------
INTENT_PATTERNS = [
    ("switching from on-prem to cloud", r"\b(on[- ]?prem|server|local)\b.*\b(cloud)\b|cloud.*on[- ]?prem"),
    ("server failure urgency", r"\b(server|hardware|raid|drive|backup).*(fail|failing|down|crash|dying)"),
    ("software consolidation", r"\b(all[- ]?in[- ]?one|consolidate|eliminat(e|ing) vendors|one (platform|system))\b"),
    ("training acceleration", r"\b(train|training|onboard|go-?live).*(one day|compressed|accelerate|fast)"),
    ("sandbox request", r"\b(sandbox|trial|test environment|playground|demo environment)\b"),
    ("budget-conscious (license avoidance)", r"\b(license|subscription|cost|price|budget|avoid paying)\b"),
    ("multi-location", r"\b(second location|multi[- ]?location|additional office|satellite)\b"),
    ("migration concerns", r"\b(migrate|migration|import|export|data loss|conversion)\b"),
]


def extract_intents(blob: str):
    tags = []
    low = (blob or "").lower()
    for name, pat in INTENT_PATTERNS:
        if re.search(pat, low, flags=re.I | re.S):
            tags.append(name)
    return sorted(set(tags))


def pick_quote_highlights(utterances, max_quotes=3):
    kws = ["cloud", "server", "license", "training", "sandbox", "migration", "data"]
    scored = []
    for u in utterances or []:
        txt = (u.get("text") or "").strip()
        if not txt:
            continue
        score = sum(1 for k in kws if k in txt.lower()) + min(len(txt), 220) / 220 * 0.2
        scored.append((score, txt))
    scored.sort(reverse=True)
    out = []
    for _, t in scored[:max_quotes]:
        t = re.sub(r"\s+", " ", t).strip()
        out.append(t[:177] + "..." if len(t) > 180 else t)
    return out


def heuristic_notes(call_obj: dict):
    raw = call_obj.get("raw_conversation") or []
    blob = " \n".join((u.get("text") or "") for u in raw)
    summary = []
    # Mentioned software/integrations
    sw_vocab = ["dentrix", "eaglesoft", "opendental", "curve", "oryx", "abeldent", "axium", "fuse", "cleardent", "dovetail", "tab32"]
    it_vocab = ["weave", "nexhealth", "pearl", "phreesia", "yapi", "dosespot", "kleer", "carecredit", "dentalintel", "revenuewell"]
    def _find_mentions(text, vocab):
        low = (text or "").lower(); return sorted({term for term in vocab if term.lower() in low})
    sw = _find_mentions(blob, sw_vocab)
    it = _find_mentions(blob, it_vocab)
    if sw:
        summary.append("Current/mentioned software: " + ", ".join(sw))
    if it:
        summary.append("Third-party tools discussed: " + ", ".join(it))
    if not summary and blob.strip():
        bits = re.split(r"(?<=[\.!\?])\s+", blob.strip())
        if bits:
            summary.append("Notable note: " + (bits[0][:160] + ("..." if len(bits[0]) > 160 else "")))
    return {
        "summary_bullets": summary[:6],
        "intent_tags": extract_intents(blob),
        "quote_highlights": pick_quote_highlights(raw),
    }


# ------------
# LLM augmentation (OpenAI; optional)
# ------------
def build_llm_prompt(call_obj: dict) -> str:
    raw = call_obj.get("raw_conversation") or []
    lines = []
    for u in raw[:200]:
        t = u.get("text") or ""
        sp = u.get("speaker") or "Speaker"
        ts = u.get("timestamp") or u.get("start_sec") or ""
        lines.append(f"[{ts}] {sp}: {t}")
    convo = "\n".join(lines)
    return (
        "You are creating concise sales intelligence from a dental software call.\n"
        "Return STRICT JSON with keys:\n"
        "- summary_bullets: array of 3-6 short bullets (who they are, current stack, pain/urgency, timeline, training, integrations)\n"
        "- intent_tags: array of short tags capturing the caller's intent/themes\n"
        "- quote_highlights: array of 2-3 short quotes (<=180 chars) that capture need/urgency/fit\n\n"
        f"Conversation (truncated):\n{convo}\n"
    )


def openai_generate(call_obj: dict):
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY not set"
    try:
        prompt = build_llm_prompt(call_obj)
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=60,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        start = content.find("{"); end = content.rfind("}")
        payload = json.loads(content[start:end+1] if start != -1 and end != -1 else content)
        return payload, "openai"
    except Exception as e:
        return None, str(e)


def generate_notes(call_obj: dict):
    notes, source = openai_generate(call_obj)
    if notes:
        return notes, source
    # fallback
    return heuristic_notes(call_obj), "fallback"


# ------------
# Output ordering
# ------------
def write_ordered_json(base: dict, notes: dict, output_path: Path):
    ordered = OrderedDict()
    # Summary fields first
    ordered["call_id"] = base.get("call_id")
    ordered["summary"] = notes.get("summary_bullets", [])
    ordered["intent_tags"] = notes.get("intent_tags", [])
    ordered["quote_highlights"] = notes.get("quote_highlights", [])
    # Transcript last
    ordered["raw_conversation"] = base.get("raw_conversation", [])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


# ------------
# Main processing
# ------------
def process_file(txt_path: Path, out_path: Path, archive_folder: Path):
    call_id = txt_path.stem
    raw = parse_transcript(txt_path)
    raw = normalize_end_times(raw)

    base = {"call_id": call_id, "raw_conversation": raw}
    notes, source = generate_notes(base)
    write_ordered_json(base, notes, out_path)

    # Archive processed .txt
    archive_folder.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(txt_path), str(archive_folder / txt_path.name))
    except Exception:
        # Cross-device safe fallback
        import shutil as _sh
        _sh.copy2(str(txt_path), str(archive_folder / txt_path.name))
        Path(txt_path).unlink(missing_ok=True)
    print(f"Wrote {out_path.name} [{source}] and archived {txt_path.name}")


def main():
    ap = argparse.ArgumentParser(description="Convert .txt transcripts to JSON with summary/tags/quotes and archive inputs.")
    ap.add_argument("--in", dest="in_dir", default=str(DEFAULT_INPUT_FOLDER), help="Input folder of .txt transcripts")
    ap.add_argument("--out", dest="out_dir", default=str(DEFAULT_OUTPUT_FOLDER), help="Output folder for .json files")
    ap.add_argument("--archive", dest="archive_dir", default=str(DEFAULT_ARCHIVE_FOLDER), help="Archive folder for processed .txt files")
    args = ap.parse_args()

    in_path = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    archive_dir = Path(args.archive_dir)

    # Single file mode
    if in_path.is_file() and in_path.suffix.lower() == ".txt":
        out_dir.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(parents=True, exist_ok=True)
        txt = in_path
        outp = out_dir / (txt.stem + ".json")
        process_file(txt, outp, archive_dir)
        return

    # Directory mode
    in_dir = in_path
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {in_dir}")
        return

    for txt in files:
        outp = out_dir / (txt.stem + ".json")
        try:
            process_file(txt, outp, archive_dir)
        except Exception as e:
            print(f"Failed {txt.name}: {e}")


if __name__ == "__main__":
    main()
