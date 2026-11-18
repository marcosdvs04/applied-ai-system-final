from src.knowledge import KnowledgeBase, Chunk, parse_chunks, tokenize


def kb():
    return KnowledgeBase.from_file()


def test_parse_chunks_reads_metadata():
    chunks = parse_chunks("## Situation: x\nkeywords: a, b\ngenre: lofi\nenergy: 0.4\nnote: hello\n")
    assert chunks[0].title == "Situation: x"
    assert chunks[0].keywords == ["a", "b"]
    assert chunks[0].meta == {"genre": "lofi", "energy": "0.4"}
    assert chunks[0].note == "hello"


def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("the gym")
    assert "gym" in tokenize("the gym")


def test_retrieves_workout_chunk_for_gym():
    top = kb().retrieve("I am going to the gym")[0]
    assert top.chunk.title == "Situation: working out"


def test_stemming_matches_studying_to_study():
    top = kb().retrieve("I am studying")[0]
    assert "studying" in top.chunk.title.lower()


def test_unrelated_query_retrieves_nothing():
    assert kb().retrieve("quantum chromodynamics lagrangian") == []


def test_empty_query_retrieves_nothing():
    assert kb().retrieve("   ") == []


def test_missing_kb_file_raises_clear_error(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        KnowledgeBase.from_file(tmp_path / "nope.md")
