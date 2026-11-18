"""
Reliability harness. Runs fixed test inputs through the assistant and checks each
against an expected property. Writes evals/results.md (a table) and prints a summary.

Run from the repo root:  python -m evals.evaluate
"""

from pathlib import Path
from statistics import mean

from src.pipeline import VibeMatchAssistant

# (input, criterion text, check function on Response)
CASES = [
    ("music for studying", "Top song is lofi", lambda r: r.recommendations[0][0]["genre"] == "lofi"),
    ("I'm at the gym", "Top song energy >= 0.9", lambda r: r.recommendations[0][0]["energy"] >= 0.9),
    ("party tonight", "Top song is edm", lambda r: r.recommendations[0][0]["genre"] == "edm"),
    ("rainy sad afternoon", "Top song is folk or classical", lambda r: r.recommendations[0][0]["genre"] in {"folk", "classical"}),
    ("night drive", "Top song is synthwave", lambda r: r.recommendations[0][0]["genre"] == "synthwave"),
    ("coffee shop vibes", "Top song is jazz", lambda r: r.recommendations[0][0]["genre"] == "jazz"),
    ("I am so angry", "Top song is metal", lambda r: r.recommendations[0][0]["genre"] == "metal"),
    ("romantic dinner date", "Top song is r&b", lambda r: r.recommendations[0][0]["genre"] == "r&b"),
    ("reggaeton para bailar", "Top song is reggaeton", lambda r: r.recommendations[0][0]["genre"] == "reggaeton"),
    ("jazz", "Handles genre-only input: top is jazz", lambda r: r.recommendations[0][0]["genre"] == "jazz"),
    ("edm with energy 1.5", "Out-of-range energy clamped to 1.0", lambda r: r.profile["target_energy"] == 1.0),
    ("k-pop", "Unknown genre flagged, no crash", lambda r: any("not in the catalog" in w for w in r.warnings)),
    ("asdf qwer zxcv", "Gibberish -> asks for clarification", lambda r: r.status == "needs_clarification"),
    ("", "Empty input -> rejected", lambda r: r.status == "rejected"),
    ("x" * 400, "Oversized input -> rejected", lambda r: r.status == "rejected"),
    ("study music, no acoustic", "Explicit 'no acoustic' overrides KB hint", lambda r: r.profile["likes_acoustic"] is False),
]


def run() -> str:
    bot = VibeMatchAssistant()
    rows, passed, confidences = [], 0, []
    for query, criterion, check in CASES:
        resp = bot.ask(query)
        try:
            ok = bool(check(resp))
        except Exception:
            ok = False
        passed += ok
        if resp.status in {"ok", "low_confidence"}:
            confidences.append(resp.confidence)
        shown = query if len(query) < 40 else query[:37] + "..."
        top = resp.recommendations[0][0]["title"] if resp.recommendations else "-"
        rows.append(f"| `{shown}` | {criterion} | {resp.status} | {resp.confidence:.2f} | {top} | {'Pass' if ok else 'FAIL'} |")

    summary = (
        f"**{passed} of {len(CASES)} checks passed.** "
        f"Average confidence on answered requests: {mean(confidences):.2f} "
        f"(min {min(confidences):.2f}, max {max(confidences):.2f})."
    )
    table = (
        "| Test input | Evaluation criterion | Status | Confidence | Top song | Result |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(rows)
    )
    return summary + "\n\n" + table + "\n"


if __name__ == "__main__":
    report = run()
    out = Path(__file__).resolve().parent / "results.md"
    out.write_text("# Evaluation results\n\n" + report, encoding="utf-8")
    print(report)
