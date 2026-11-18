"""
Guardrails: input validation, profile sanitizing and output checks.

Every function returns data plus human-readable warnings instead of raising,
so the pipeline can keep going safely and tell the user what was changed.
"""

from typing import Dict, List, Tuple

MAX_QUERY_CHARS = 300
MIN_CONFIDENCE_TO_RECOMMEND = 0.30   # below this the system asks for clarification
LOW_CONFIDENCE_WARNING = 0.55        # below this we recommend but flag uncertainty


def validate_query(query) -> Tuple[str, List[str]]:
    """Return (clean_query, errors). A non-empty errors list means: reject."""
    if not isinstance(query, str):
        return "", ["Query must be text."]
    clean = " ".join(query.split())  # collapse whitespace/newlines
    if not clean:
        return "", ["Query is empty. Describe a situation, mood or genre (e.g. 'music for studying')."]
    if len(clean) > MAX_QUERY_CHARS:
        return "", [f"Query is too long ({len(clean)} chars, max {MAX_QUERY_CHARS})."]
    return clean, []


def sanitize_profile(profile: Dict, genres: set, moods: set) -> Tuple[Dict, List[str]]:
    """Clamp numeric values to valid ranges and flag preferences the catalog can't satisfy."""
    warnings: List[str] = []
    clean = dict(profile)

    energy = clean.get("target_energy", 0.5)
    try:
        energy = float(energy)
    except (TypeError, ValueError):
        warnings.append("Energy was not a number; using 0.5.")
        energy = 0.5
    if energy < 0.0 or energy > 1.0:
        warnings.append(f"Energy {energy} is outside 0-1; clamped.")
        energy = min(1.0, max(0.0, energy))
    clean["target_energy"] = energy

    clean["likes_acoustic"] = bool(clean.get("likes_acoustic", False))

    genre = str(clean.get("favorite_genre", "")).lower()
    mood = str(clean.get("favorite_mood", "")).lower()
    clean["favorite_genre"], clean["favorite_mood"] = genre, mood
    if genre and genre not in genres:
        warnings.append(f"Genre '{genre}' is not in the catalog, so it can't earn genre points.")
    if mood and mood not in moods:
        warnings.append(f"Mood '{mood}' is not in the catalog, so it can't earn mood points.")
    return clean, warnings


def check_recommendations(results: list, k: int, max_score: float) -> List[str]:
    """Sanity-check the ranked output. Returns a list of problems (empty = healthy)."""
    problems: List[str] = []
    if not results:
        return ["No recommendations were produced."]
    if len(results) > k:
        problems.append("More results than requested.")
    scores = [r[1] for r in results]
    if any(s < 0 or s > max_score + 1e-9 for s in scores):
        problems.append("A score fell outside the valid range.")
    if scores != sorted(scores, reverse=True):
        problems.append("Results are not sorted by score.")
    return problems
