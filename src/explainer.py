"""
Grounded explanations (the "G" in RAG).

Offline mode (default, no API key): builds the explanation only from
  - the recommender's score reasons, and
  - the `note:` text of the retrieved knowledge-base chunks, cited as [KB: title].

Optional LLM mode (GEMINI_API_KEY set and use_llm=True): asks Gemini to rewrite the
same facts in a friendlier tone. The reply is verified (it must not mention songs
that were not recommended); if the call fails or the check fails, the offline
explanation is used. The LLM can never change the ranking.
"""

import json
import logging
import os
import urllib.request
from typing import List

from src.knowledge import Retrieved

logger = logging.getLogger("vibematch")


def offline_summary(query: str, retrieved: List[Retrieved], recs: list) -> str:
    """One paragraph tying the request, the retrieved knowledge and the top pick together."""
    parts = []
    if retrieved:
        r = retrieved[0]
        if r.chunk.note:
            parts.append(f"{r.chunk.note} [KB: {r.chunk.title}]")
    if recs:
        top, score, _ = recs[0]
        parts.append(f"Top pick for \"{query}\": {top['title']} by {top['artist']} (score {score:.2f}).")
    return " ".join(parts)


def llm_rewrite(summary: str, recs: list) -> str:
    """Try to rewrite `summary` with Gemini. Returns the original text on any problem."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return summary
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    titles = [song["title"] for song, _, _ in recs]
    prompt = (
        "Rewrite the text below in two friendly sentences. Use ONLY facts in the text. "
        "Do not add songs, artists or claims that are not in it.\n\nTEXT:\n" + summary
    )
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = json.load(resp)["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as exc:  # network, quota, bad key, bad JSON...
        logger.warning("LLM rewrite failed (%s); using offline explanation", type(exc).__name__)
        return summary
    if not text or not _grounded(text, summary, titles):
        logger.warning("LLM rewrite failed grounding check; using offline explanation")
        return summary
    return text


def _grounded(text: str, source: str, titles: List[str]) -> bool:
    """Reject rewrites that introduce quoted titles not present in the source text."""
    import re
    quoted = re.findall(r'"([^"]+)"', text)
    return all(q in source or q in titles for q in quoted)
