import pytest

from src.pipeline import VibeMatchAssistant


@pytest.fixture(scope="module")
def bot():
    return VibeMatchAssistant()


def test_study_request_returns_lofi_first(bot):
    r = bot.ask("music for studying")
    assert r.status == "ok"
    assert r.recommendations[0][0]["genre"] == "lofi"


def test_retrieval_changes_the_recommendations(bot):
    """RAG is integrated: different retrieved context -> different profile -> different top song."""
    study = bot.ask("music for studying").recommendations[0][0]["title"]
    gym = bot.ask("music for the gym").recommendations[0][0]["title"]
    assert study != gym


def test_explicit_words_override_retrieved_hints(bot):
    r = bot.ask("studying but I want metal")
    assert r.profile["favorite_genre"] == "metal"
    assert r.profile["favorite_mood"] == "focused"  # still taken from the retrieved chunk


def test_explanation_cites_knowledge_base(bot):
    assert "[KB:" in bot.ask("night drive").explanation


def test_gibberish_asks_for_clarification(bot):
    r = bot.ask("zzzz qqqq")
    assert r.status == "needs_clarification" and not r.recommendations


def test_empty_input_is_rejected_not_crashed(bot):
    assert bot.ask("").status == "rejected"
    assert bot.ask(None).status == "rejected"


def test_out_of_range_energy_is_clamped(bot):
    r = bot.ask("edm with energy 1.5")
    assert r.profile["target_energy"] == 1.0
    assert any("clamped" in w for w in r.warnings)


def test_k_is_respected_and_sorted(bot):
    r = bot.ask("workout", k=3)
    scores = [s for _, s, _ in r.recommendations]
    assert len(scores) == 3 and scores == sorted(scores, reverse=True)


def test_llm_mode_without_key_falls_back(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = VibeMatchAssistant(use_llm=True).ask("coffee shop")
    assert r.status in {"ok", "low_confidence"} and r.explanation


def test_log_file_is_written(bot, tmp_path):
    from src.pipeline import LOG_PATH
    bot.ask("party")
    assert LOG_PATH.exists() and "party" in LOG_PATH.read_text()


def test_uncataloged_genre_is_flagged_not_ignored(bot):
    r = bot.ask("k-pop")
    assert r.profile["favorite_genre"] == "k-pop"
    assert any("not in the catalog" in w for w in r.warnings)
