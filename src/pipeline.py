"""
VibeMatch assistant: the full pipeline.

  query -> validate -> retrieve (RAG) -> build profile -> sanitize -> recommend
        -> check output -> grounded explanation -> log

Every request is logged to logs/vibematch.log. Nothing here raises for bad user
input: problems come back in `Response.status` and `Response.warnings`.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src import guardrails
from src.explainer import llm_rewrite, offline_summary
from src.knowledge import KnowledgeBase
from src.profile_builder import ProfileResult, build_profile
from src.recommender import MAX_SCORE, load_songs, recommend_songs

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "vibematch.log"
SONGS_PATH = ROOT / "data" / "songs.csv"

logger = logging.getLogger("vibematch")


def setup_logging(path: Path = LOG_PATH) -> None:
    """Log to a file (not the console) once; safe to call repeatedly."""
    if logger.handlers:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class Response:
    status: str                      # ok | low_confidence | needs_clarification | rejected
    query: str
    confidence: float = 0.0
    profile: Dict = field(default_factory=dict)
    recommendations: List[tuple] = field(default_factory=list)   # (song, score, reasons)
    explanation: str = ""
    evidence: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class VibeMatchAssistant:
    def __init__(self, songs_path: Path = SONGS_PATH, kb: Optional[KnowledgeBase] = None, use_llm: bool = False):
        setup_logging()
        self.songs = load_songs(str(songs_path))
        self.kb = kb or KnowledgeBase.from_file()
        self.use_llm = use_llm
        self.genres = {s["genre"].lower() for s in self.songs}
        self.moods = {s["mood"].lower() for s in self.songs}

    def ask(self, query, k: int = 5) -> Response:
        clean, errors = guardrails.validate_query(query)
        if errors:
            logger.warning("REJECTED query=%r errors=%s", str(query)[:80], errors)
            return Response("rejected", str(query)[:80] if isinstance(query, str) else "", warnings=errors)

        try:
            result: ProfileResult = build_profile(clean, self.kb, self.genres, self.moods)
            profile, warnings = guardrails.sanitize_profile(result.profile, self.genres, self.moods)
        except Exception:  # last-resort safety net; never crash the caller
            logger.exception("Profile building crashed for query=%r", clean)
            return Response("rejected", clean, warnings=["Internal error while reading the request."])

        resp = Response("ok", clean, result.confidence, profile, evidence=result.evidence, warnings=warnings)

        if result.confidence < guardrails.MIN_CONFIDENCE_TO_RECOMMEND:
            resp.status = "needs_clarification"
            resp.warnings.append(
                "I couldn't tell what you want. Try a situation (studying, workout, night drive), "
                "a genre (jazz, edm) or a mood (sad, happy)."
            )
            logger.info("CLARIFY query=%r confidence=%.2f", clean, result.confidence)
            return resp

        recs = recommend_songs(profile, self.songs, k=k)
        problems = guardrails.check_recommendations(recs, k, MAX_SCORE)
        if problems:
            resp.status = "rejected"
            resp.warnings.extend(problems)
            logger.error("OUTPUT CHECK FAILED query=%r problems=%s", clean, problems)
            return resp

        resp.recommendations = recs
        summary = offline_summary(clean, result.retrieved, recs)
        resp.explanation = llm_rewrite(summary, recs) if self.use_llm else summary

        if result.confidence < guardrails.LOW_CONFIDENCE_WARNING:
            resp.status = "low_confidence"
            resp.warnings.append("Low confidence: some preferences were guessed. Add a genre or mood to sharpen results.")

        logger.info(
            "OK query=%r status=%s confidence=%.2f profile=%s top=%r",
            clean, resp.status, resp.confidence, profile, recs[0][0]["title"],
        )
        return resp


def format_response(resp: Response) -> str:
    """Plain-text rendering used by the CLI and by the README sample runs."""
    lines = [f'Query: "{resp.query}"', f"Status: {resp.status}   Confidence: {resp.confidence:.2f}"]
    if resp.profile:
        p = resp.profile
        lines.append(
            f"Profile: genre={p['favorite_genre'] or '?'}, mood={p['favorite_mood'] or '?'}, "
            f"energy={p['target_energy']:.2f}, acoustic={p['likes_acoustic']}"
        )
    for ev in resp.evidence:
        lines.append(f"  evidence: {ev}")
    if resp.explanation:
        lines.append(f"Why: {resp.explanation}")
    for i, (song, score, reasons) in enumerate(resp.recommendations, 1):
        lines.append(f"{i}. {song['title']} - {song['artist']} ({song['genre']}/{song['mood']}) score {score:.2f}/{MAX_SCORE:.1f}")
        lines.append(f"     because: {reasons}")
    for w in resp.warnings:
        lines.append(f"! {w}")
    return "\n".join(lines)
