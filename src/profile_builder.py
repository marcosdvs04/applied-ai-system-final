"""
Turns a free-text request into a UserProfile dict for the recommender.

Evidence sources, strongest first:
  1. Explicit words in the query ("jazz", "sad", "energy 0.8", "acoustic")
  2. Retrieved knowledge-base chunks (RAG): "gym" -> workout chunk -> pop / intense / 0.9
  3. Defaults (energy 0.5, acoustic False) -> no evidence, lowers confidence

The retrieved chunks actively decide the profile, so retrieval changes the
recommendations; it is not decoration printed next to a normal answer.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.knowledge import KnowledgeBase, Retrieved

# Words people use that map to catalog values.
GENRE_ALIASES = {
    "lo-fi": "lofi", "lo fi": "lofi", "rap": "hip hop", "hip-hop": "hip hop", "hiphop": "hip hop",
    "rnb": "r&b", "r and b": "r&b", "electronic": "edm", "techno": "edm", "house": "edm",
    "indie": "indie pop", "latin": "reggaeton", "orchestral": "classical", "piano": "classical",
}
MOOD_ALIASES = {
    "calm": "chill", "peaceful": "chill", "upbeat": "happy", "cheerful": "happy", "joyful": "happy",
    "aggressive": "angry", "furious": "angry", "mad": "angry", "gloomy": "sad", "depressed": "sad",
    "sentimental": "nostalgic", "dreamy": "moody", "hype": "energetic", "pumped": "energetic",
    "concentrated": "focused", "love": "romantic", "mellow": "relaxed", "cozy": "relaxed",
}
# Real genres the 18-song catalog does not cover. Recognizing them lets the system warn
# the user instead of silently ignoring the word (found by the evaluation harness).
UNCATALOGED_GENRES = {"k-pop", "kpop", "punk", "gospel", "reggae", "blues", "disco", "funk", "opera", "salsa", "bachata"}
HIGH_ENERGY_PHRASES = ["high energy", "high-energy", "very energetic", "loud", "intense", "hype", "pumped", "fast"]
LOW_ENERGY_PHRASES = ["low energy", "low-energy", "quiet", "soft", "mellow", "calm", "slow", "gentle", "sleep"]

# Weight of each field in the confidence score (sums to 1.0).
FIELD_WEIGHTS = {"genre": 0.30, "mood": 0.30, "energy": 0.25, "acoustic": 0.15}


@dataclass
class ProfileResult:
    profile: Dict
    confidence: float
    evidence: List[str] = field(default_factory=list)        # human-readable trail
    sources: Dict[str, str] = field(default_factory=dict)    # field -> explicit/retrieved/default
    retrieved: List[Retrieved] = field(default_factory=list)


def _find_vocab(query_l: str, vocab: Set[str], aliases: Dict[str, str]) -> Optional[str]:
    """Return the first catalog value (or alias) mentioned as a whole word."""
    cleaned = re.sub(r"[^a-z0-9&\- ]", " ", query_l)
    padded = " " + cleaned + " "
    # longest first so "indie pop" wins over "pop"
    candidates = sorted(set(vocab) | set(aliases), key=len, reverse=True)
    for cand in candidates:
        if f" {cand} " in padded:
            return aliases.get(cand, cand)
    return None


def _parse_energy(query_l: str) -> Optional[float]:
    m = re.search(r"energy\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*(%?)", query_l)
    if m:
        value = float(m.group(1))
        return value / 100 if m.group(2) or value > 10 else value  # 80% or 80 -> 0.8; 1.5 stays 1.5
    if any(p in query_l for p in HIGH_ENERGY_PHRASES):
        return 0.9
    if any(p in query_l for p in LOW_ENERGY_PHRASES):
        return 0.3
    return None


def _parse_acoustic(query_l: str) -> Optional[bool]:
    if re.search(r"\b(no|not|without|non)[ -]acoustic\b|\bdon'?t (like|want) acoustic\b", query_l):
        return False
    if re.search(r"\b(acoustic|unplugged)\b", query_l):
        return True
    if re.search(r"\b(electronic|synth|synthetic)\b", query_l):
        return False
    return None


def build_profile(query: str, kb: KnowledgeBase, genres: Set[str], moods: Set[str]) -> ProfileResult:
    q = query.lower()
    evidence: List[str] = []
    sources: Dict[str, str] = {}
    strength: Dict[str, float] = {}
    profile: Dict = {}

    # --- 1. Retrieval (RAG): find the situation/genre chunks that match the request.
    retrieved = kb.retrieve(query, k=2)
    situation = next((r for r in retrieved if r.chunk.kind == "situation"), None)
    for r in retrieved:
        evidence.append(f"retrieved '{r.chunk.title}' (score {r.score:.1f})")

    # --- 2. Explicit words always beat retrieved hints.
    genre = _find_vocab(q, genres, GENRE_ALIASES) or _find_vocab(q, UNCATALOGED_GENRES, {})
    mood = _find_vocab(q, moods, MOOD_ALIASES)
    energy = _parse_energy(q)
    acoustic = _parse_acoustic(q)

    def fill(name: str, explicit, meta_key: str, convert=lambda v: v):
        if explicit is not None:
            sources[name], strength[name] = "explicit", 1.0
            evidence.append(f"{name}: '{explicit}' stated in the request")
            return explicit
        for r in ([situation] if situation else []) + [x for x in retrieved if x is not situation]:
            if meta_key in r.chunk.meta:
                sources[name], strength[name] = "retrieved", r.strength
                value = convert(r.chunk.meta[meta_key])
                evidence.append(f"{name}: '{value}' from '{r.chunk.title}'")
                return value
        sources[name], strength[name] = "default", 0.0
        return None

    profile["favorite_genre"] = fill("genre", genre, "genre") or ""
    profile["favorite_mood"] = fill("mood", mood, "mood") or ""
    e = fill("energy", energy, "energy", float)
    profile["target_energy"] = 0.5 if e is None else e
    a = fill("acoustic", acoustic, "acoustic", lambda v: v.strip().lower() in {"yes", "true"})
    profile["likes_acoustic"] = False if a is None else a

    # A genre the catalog can't satisfy is not usable evidence, however clearly it was stated.
    if profile["favorite_genre"] and profile["favorite_genre"] not in genres:
        strength["genre"] = 0.0

    confidence = round(sum(FIELD_WEIGHTS[f] * strength[f] for f in FIELD_WEIGHTS), 2)
    return ProfileResult(profile, confidence, evidence, sources, retrieved)
