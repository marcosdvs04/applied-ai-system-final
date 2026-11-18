from src import guardrails as g

GENRES, MOODS = {"pop", "lofi"}, {"happy", "chill"}


def test_empty_query_rejected():
    _, errors = g.validate_query("   ")
    assert errors


def test_non_string_query_rejected():
    _, errors = g.validate_query(None)
    assert errors


def test_long_query_rejected():
    _, errors = g.validate_query("a" * (g.MAX_QUERY_CHARS + 1))
    assert errors


def test_whitespace_is_collapsed():
    clean, errors = g.validate_query("  study \n  music ")
    assert clean == "study music" and not errors


def test_energy_is_clamped_with_warning():
    profile, warnings = g.sanitize_profile({"target_energy": 1.5, "favorite_genre": "pop", "favorite_mood": "happy"}, GENRES, MOODS)
    assert profile["target_energy"] == 1.0
    assert any("clamped" in w for w in warnings)


def test_non_numeric_energy_falls_back():
    profile, warnings = g.sanitize_profile({"target_energy": "loud"}, GENRES, MOODS)
    assert profile["target_energy"] == 0.5 and warnings


def test_unknown_genre_is_flagged():
    _, warnings = g.sanitize_profile({"favorite_genre": "k-pop", "favorite_mood": "happy", "target_energy": 0.5}, GENRES, MOODS)
    assert any("k-pop" in w for w in warnings)


def test_check_recommendations_detects_bad_order():
    results = [({}, 1.0, ""), ({}, 2.0, "")]
    assert "not sorted" in " ".join(g.check_recommendations(results, 5, 6.5))


def test_check_recommendations_detects_empty():
    assert g.check_recommendations([], 5, 6.5)
