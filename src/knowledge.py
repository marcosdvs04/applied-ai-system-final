"""
Knowledge base + retriever (the "R" in RAG).

The knowledge base is a Markdown file where every `## ` section is one chunk.
A chunk has a title, optional metadata lines (keywords, genre, mood, energy,
acoustic) and a human-readable `note:` that is cited later in the answer.

Retrieval is keyword based on purpose: it is transparent, needs no API key and
is easy to test. A chunk is only returned when at least one query word hits
its title or keywords, which doubles as a guardrail against irrelevant context.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge" / "music_knowledge.md"

STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "with", "me", "my",
    "i", "some", "something", "music", "songs", "song", "play", "give", "want", "need",
    "like", "that", "this", "is", "it", "be", "can", "you", "please", "playlist", "while",
    "when", "feel", "feeling", "good", "listen", "listening", "something",
}

KEYWORD_WEIGHT = 2.0
TITLE_WEIGHT = 1.0


def tokenize(text: str) -> List[str]:
    """Lowercase, split into words (keeps r&b / lo-fi style tokens), drop stopwords."""
    words = re.findall(r"[a-z0-9&'\-]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def _stem(word: str) -> str:
    """Very small stemmer so 'studying' ~ 'study' and 'drives' ~ 'drive'."""
    for suffix in ("ing", "es", "s", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


@dataclass
class Chunk:
    title: str
    keywords: List[str] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def kind(self) -> str:
        return self.title.split(":", 1)[0].strip().lower()  # "situation" or "genre"


@dataclass
class Retrieved:
    chunk: Chunk
    score: float        # raw overlap score
    strength: float     # 0..1, used as evidence strength for confidence


class KnowledgeBase:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._index = [self._build_terms(c) for c in chunks]

    # ---------- loading ----------
    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "KnowledgeBase":
        path = Path(path) if path else DEFAULT_KB_PATH
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")
        return cls(parse_chunks(path.read_text(encoding="utf-8")))

    @staticmethod
    def _build_terms(chunk: Chunk) -> Dict[str, float]:
        terms: Dict[str, float] = {}
        for tok in tokenize(chunk.title):
            terms[_stem(tok)] = max(terms.get(_stem(tok), 0), TITLE_WEIGHT)
        for kw in chunk.keywords:
            for tok in tokenize(kw):
                terms[_stem(tok)] = max(terms.get(_stem(tok), 0), KEYWORD_WEIGHT)
        return terms

    # ---------- retrieval ----------
    def retrieve(self, query: str, k: int = 2, min_score: float = KEYWORD_WEIGHT) -> List[Retrieved]:
        """Return up to k chunks ranked by weighted word overlap.

        Chunks scoring below `min_score` (i.e. no real keyword hit) are dropped,
        so an off-topic question retrieves nothing instead of something random.
        """
        q_tokens = {_stem(t) for t in tokenize(query)}
        if not q_tokens:
            return []
        results: List[Retrieved] = []
        for chunk, terms in zip(self.chunks, self._index):
            score = sum(terms[t] for t in q_tokens if t in terms)
            if score >= min_score:
                strength = min(1.0, 0.5 + 0.25 * score / KEYWORD_WEIGHT)
                results.append(Retrieved(chunk, score, strength))
        results.sort(key=lambda r: (r.score, r.chunk.kind == "situation"), reverse=True)
        return results[:k]


def parse_chunks(markdown: str) -> List[Chunk]:
    """Split the Markdown file on `## ` headings and read the metadata lines."""
    chunks: List[Chunk] = []
    current: Optional[Chunk] = None
    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            current = Chunk(title=line[3:].strip())
            chunks.append(current)
        elif current is not None and ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            key, value = key.strip().lower(), value.strip()
            if key == "keywords":
                current.keywords = [k.strip() for k in value.split(",") if k.strip()]
            elif key == "note":
                current.note = value
            elif key in {"genre", "mood", "energy", "acoustic"}:
                current.meta[key] = value
    return chunks
