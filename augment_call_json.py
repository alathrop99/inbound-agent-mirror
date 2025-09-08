# augment_call_json.py
import json
import re
import sys
from pathlib import Path
from datetime import timedelta

# --- Configurable keyword buckets (seed these; expand as you wish) ---
SOFTWARE = [
    "dentrix", "eaglesoft", "opendental", "curve", "oryx", "abeldent",
    "axiUm", "doctorsgateway", "fuse", "cleardent", "dovetail", "tab32"
]
INTEGRATIONS = [
    "weave", "nexhealth", "pearl", "phreesia", "yapi", "dosespot",
    "kleer", "rectangle", "carecredit", "dentalintel", "revenuewell"
]
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

# --- Helpers ---
def as_secs(ts_like):
    # Accepts "7:32" or seconds (int/float) and returns seconds as float
    if isinstance(ts_like, (int, float)):
        return float(ts_like)
    if isinstance(ts_like, str) and re.match(r"^\d+:\d{2}$", ts_like.strip()):
        m, s = ts_like.split(":")
        return int(m) * 60 + int(s)
    try:
        return float(ts_like)
    except Exception:
        return None

def sec_to_mmss(x):
    return str(timedelta(seconds=int(max(0, x))))[2:] if x is not None else None

def normalize_end_times(utterances, epsilon=0.05):
    """Set each utterance end_sec to next utterance start_sec - epsilon.
       Keeps interleaving intact when someone interrupts."""
    utts = sorted(utterances, key=lambda u: u.get("start_sec", 0))
    for i, u in enumerate(utts):
        if "start_sec" not in u:
            u["start_sec"] = as_secs(u.get("start_time")) or 0.0
        if i < len(utts) - 1:
            next_start = utts[i+1].get("start_sec", 0.0)
            u["end_sec"] = max(u.get("start_sec", 0.0), next_start - epsilon)
        else:
            # Last utterance: default to +3s if unknown
            u["end_sec"] = u.get("end_sec") or (u.get("start_sec", 0.0) + 3.0)
    return utts

def find_mentions(text, vocab):
    found = set()
    low = text.lower()
    for term in vocab:
        if term.lower() in low:
            found.add(term)
    return sorted(found)

def extract_intents(full_text):
    tags = []
    low = full_text.lower()
    for name, pat in INTENT_PATTERNS:
        if re.search(pat, low, flags=re.I|re.S):
            tags.append(name)
    return sorted(set(tags))

def pick_quote_highlights(utterances, max_quotes=3):
    # Heuristic: keep customer quotes that mention software, urgency, or “cloud”
    kws = SOFTWARE + ["cloud", "server", "license", "training", "sandbox"]
    scored = []
    for u in utterances:
        speaker = (u.get("speaker") or "").lower()
        txt = (u.get("text") or u.get("content") or "").strip()
        if not txt:
            continue
        # prefer the *customer* side if you have speaker labels (tweak as needed)
        customer_bias = 1.2 if speaker and any(k in speaker for k in ["dr", "caller", "prospect", "owner"]) else 1.0
        score = sum(w in txt.lower() for w in kws) * customer_bias + min(len(txt), 220)/220*0.2
        scored.append((score, txt))
    scored.sort(reverse=True, key=lambda x: x[0])
    quotes = []
    for _, q in scored[:max_quotes]:
        # keep to ~180 chars
        q = re.sub(r"\s+", " ", q).strip()
        if len(q) > 180:
            q = q[:177] + "…"
        quotes.append(q)
    return quotes

def make_summary(call):
    # Build a concise bullet summary using available fields + utterances
    bullets = []
    pn = call.get("practice_name") or "Unknown practice"
    caller = call.get("caller_name") or "Unknown caller"
    loc = call.get("location") or "Unknown location"
    ptype = call.get("practice_type") or ""
    bullets.append(f"{pn} ({ptype}) — Caller: {caller} — Location: {loc}")

    # software mentions
    all_text = []
    raw = call.get("raw_conversation") or []
    for u in raw:
        all_text.append(u.get("text") or u.get("content") or "")
    blob = " \n".join(all_text)

    software_found = find_mentions(blob, SOFTWARE)
    integrations_found = find_mentions(blob, INTEGRATIONS)
    if software_found:
        bullets.append("Current/mentioned software: " + ", ".join(software_found))
    if integrations_found:
        bullets.append("Third-party tools discussed: " + ", ".join(integrations_found))

    # migration/training timing if present
    tline = call.get("transition_timeline") or {}
    urgency = tline.get("urgency")
    closing = tline.get("closing_on_second_practice")
    if urgency or closing:
        bullets.append("Timeline: " + "; ".join(filter(None, [
            f"Urgency {urgency}" if urgency else None,
            f"Second practice closing {closing}" if closing else None
        ])))

    train = call.get("training_plan") or {}
    if train.get("actual") or train.get("standard"):
        bullets.append("Training: " + (train.get("actual") or train.get("standard")))

    # fallback: pull 1 high-signal utterance if bullets are sparse
    if len(bullets) < 4 and blob.strip():
        sample = re.split(r"(?<=[\.\!\?])\s+", blob.strip())
        if sample:
            bullets.append("Notable note: " + sample[0][:160] + ("…" if len(sample[0]) > 160 else ""))

    return bullets[:6]

def augment(in_path: Path):
    with in_path.open("r", encoding="utf-8") as f:
        call = json.load(f)

    # Normalize utterance times if present
    raw = call.get("raw_conversation") or []
    # Try to compute start_sec if missing (handles "mm:ss | Speaker" style too)
    for u in raw:
        if "start_sec" not in u:
            # try start_time, else try to parse from a "timestamp" or "time" string
            for key in ("start_time", "time", "timestamp"):
                if key in u and u[key]:
                    s = as_secs(u[key])
                    if s is not None:
                        u["start_sec"] = s
                        break

    call["raw_conversation"] = normalize_end_times(raw)

    # Build text blob for tagging
    blob = " \n".join((u.get("text") or u.get("content") or "") for u in call["raw_conversation"])

    # Intent tags (regex-based)
    intent_tags = extract_intents(blob)

    # Quote highlights
    quote_highlights = pick_quote_highlights(call["raw_conversation"])

    # Summary bullets
    summary_bullets = make_summary(call)

    call["quote_highlights"] = quote_highlights
    call["intent_tags"] = intent_tags
    call["summary"] = summary_bullets

    out_path = in_path.with_suffix(".augmented.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(call, f, ensure_ascii=False, indent=2)
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python augment_call_json.py path/to/call.json")
        sys.exit(1)
    path = Path(sys.argv[1])
    out = augment(path)
    print(f"✓ Augmented JSON written to: {out}")
