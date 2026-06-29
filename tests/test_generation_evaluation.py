from evaluation.scripts.generation_eval import (
    binary_to_ox,
    build_row_note,
    most_frequent_element,
    output_fieldnames,
    similarity_to_ox,
)


def test_similarity_to_ox_uses_threshold_four() -> None:
    assert similarity_to_ox(4) == "O"
    assert similarity_to_ox(5) == "O"
    assert similarity_to_ox(3) == "X"


def test_binary_to_ox_accepts_only_one() -> None:
    assert binary_to_ox(1) == "O"
    assert binary_to_ox(0) == "X"
    assert binary_to_ox(-1) == "X"


def test_most_frequent_element_prefers_x_on_tie() -> None:
    assert most_frequent_element(["O", "X"]) == "X"
    assert most_frequent_element(["O", "O", "X", "X"]) == "X"
    assert most_frequent_element(["O", "O", "X"]) == "O"


def test_output_fieldnames_include_generation_and_judge_models() -> None:
    fields = output_fieldnames()
    assert "generation_provider" in fields
    assert "generation_model" in fields
    assert "judge_model" in fields


def test_build_row_note_includes_models_and_retrieved_ids() -> None:
    note = build_row_note(
        generated_answer="응시수수료는 기준에 따라 반환됩니다.",
        retrieved_answer_nos="12|15",
        generation_provider="gemini",
        generation_model="gemini-3.5-flash",
        judge_model="meta-llama/llama-3.3-70b-instruct:free",
        judge_scores=type("JudgeScoresStub", (), {
            "similarity": 5,
            "groundedness": 2,
            "correctness": 1,
            "completeness": 1,
            "hallucination": 1,
            "overall": "pass",
        })(),
        question_note="sample",
    )
    assert "generation_provider=gemini" in note
    assert "generation_model=gemini-3.5-flash" in note
    assert "judge_model=meta-llama/llama-3.3-70b-instruct:free" in note
    assert "retrieved_answer_nos=12|15" in note
