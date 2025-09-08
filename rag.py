import os
import re
import json
import faiss
import glob
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz

DATA_DIR = "knowledge"
ALIAS_FILE = os.path.join(DATA_DIR, "aliases.json")
INDEX_FILE = os.path.join(DATA_DIR, "kb.index")
META_FILE  = os.path.join(DATA_DIR, "kb_meta.npy")
EMB_MODEL = "all-MiniLM-L6-v2"

# Very light sentence-ish splitter
_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+|\n{2,}')

def iter_docs() -> List[Tuple[str, str]]:
    """Yield (source_path, text) for all .txt/.md under knowledge/"""
    for path in glob.glob(os.path.join(DATA_DIR, "**", "*.*"), recursive=True):
        if path.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read().strip()
                if txt:
                    yield (os.path.relpath(path, DATA_DIR), txt)

def chunk(text: str, max_chars: int = 280) -> List[str]:
    """Split into small chunks; avoid mega-lines."""
    raw = [s.strip() for s in _SPLIT_RE.split(text) if s.strip()]
    out = []
    for s in raw:
        if len(s) <= max_chars:
            out.append(s)
        else:
            # simple fixed split for long bullets/paras
            for i in range(0, len(s), max_chars):
                out.append(s[i:i+max_chars])
    return out

class SimpleRAG:
    def __init__(self, facts_file: str = None, model_name: str = EMB_MODEL):
        # facts_file kept for backwards compat; we now scan all files
        self.model = SentenceTransformer(model_name)
        self.aliases: Dict[str, List[str]] = self._load_aliases()
        self.texts: List[str] = []
        self.meta: List[Tuple[str, int]] = []  # (source_path, line_index)
        self.index = None
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self._load_persisted()
        else:
            self.rebuild()

    def _load_aliases(self) -> Dict[str, List[str]]:
        if os.path.exists(ALIAS_FILE):
            with open(ALIAS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        # Defaults — extend as needed
        return {
            "weave": ["weave", "weve", "weeve", "wave", "leave", "weve communications"],
            "dentrix": ["dentrix", "dent ricks", "den tricks"],
            "eaglesoft": ["eaglesoft", "eagle soft"],
            "open dental": ["open dental", "opendental", "open dentist"]
        }

    def _persist(self, embeddings: np.ndarray):
        faiss.write_index(self.index, INDEX_FILE)
        np.save(META_FILE, np.array(self.meta, dtype=object))

    def _load_persisted(self):
        self.index = faiss.read_index(INDEX_FILE)
        self.meta = list(np.load(META_FILE, allow_pickle=True))
        # Rebuild self.texts from meta+files
        self.texts = []
        for src, line_idx in self.meta:
            # re-open source to get the text line; but we’ll also cache in memory
            with open(os.path.join(DATA_DIR, src), "r", encoding="utf-8", errors="ignore") as f:
                lines = chunk(f.read())
                self.texts.append(lines[line_idx] if 0 <= line_idx < len(lines) else "")
        # NOTE: We don’t store embeddings on disk to keep things simple.

    def rebuild(self):
        """(Re)build the FAISS index from all .txt/.md files."""
        self.texts = []
        self.meta = []
        pairs = []
        for src, raw in iter_docs():
            lines = chunk(raw)
            for i, s in enumerate(lines):
                if s:
                    self.texts.append(s)
                    self.meta.append((src, i))
                    pairs.append(s)
        if not pairs:
            self.index = None
            return
        embs = self.model.encode(pairs, convert_to_numpy=True, show_progress_bar=False)
        dim = embs.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embs)
        self._persist(embs)

    # ---------- Query helpers ----------

    def _expand_query(self, q: str) -> str:
        """
        Expand common entities by fuzzy-correcting to our alias dictionary.
        If a token is close to a known alias, replace it with the canonical key.
        """
        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", q.lower())
        canonical = list(self.aliases.keys())
        out = tokens[:]
        for i, tok in enumerate(tokens):
            # find best match among all alias forms
            all_forms = []
            for canon, forms in self.aliases.items():
                for f in forms + [canon]:
                    all_forms.append((canon, f))
            best = process.extractOne(tok, [f for _, f in all_forms], scorer=fuzz.token_sort_ratio)
            if best and best[1] >= 85:  # similarity score
                # map back to the canonical
                for canon, form in all_forms:
                    if form == best[0]:
                        out[i] = canon
                        break
        return " ".join(out) if out else q

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, str]]:
        """
        Return top-k (text, source) tuples.
        """
        if not self.index or not self.texts:
            return []
        eq = self._expand_query(query)
        q_emb = self.model.encode([eq], convert_to_numpy=True, show_progress_bar=False)
        D, I = self.index.search(q_emb, k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.texts):
                src, _line = self.meta[idx]
                results.append((self.texts[idx], src))
        return results

# Quick CLI test
if __name__ == "__main__":
    rag = SimpleRAG()
    q = "Do you integrate with leave?"
    print("Query:", q)
    hits = rag.retrieve(q, k=3)
    for t, s in hits:
        print(" -", t, f"({s})")
