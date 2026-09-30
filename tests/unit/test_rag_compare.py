import pytest

from evaluation.rag.run import compare_records

METADATA = {
    "dataset": {
        "path": "data/eval/rag.jsonl",
        "versions": ["rag-v1"],
        "splits": ["dev"],
        "n_cases": 1,
    },
    "models": {"generation": "gpt-test", "grader": "grader-test"},
    "pricing": {"version": "2026-09"},
    "params": {"top_k": 4, "chunking_version": "v1", "generation_prompt": "faq-v1"},
}
BASELINE = [
    {
        "case_id": "rag-1",
        "answer": "I do not know",
        "grounded": False,
        "retrieved_ids": ["wrong#0"],
        "duration_ms": 100.0,
    }
]
CRAG = [
    {
        "case_id": "rag-1",
        "answer": "Hours are 8:30",
        "grounded": True,
        "retrieved_ids": ["hours#0"],
        "duration_ms": 350.0,
    }
]


def test_comparison_reports_paired_changes_and_latency_delta() -> None:
    result = compare_records(METADATA, METADATA, BASELINE, CRAG)

    assert result["changed_answers"] == 1
    assert result["changed_retrievals"] == 1
    assert result["paired_cases"][0]["grounded_changed"]
    assert result["paired_cases"][0]["latency_delta_ms"] == 250.0
    assert "experimental" in result["recommendation"]


@pytest.mark.parametrize(
    ("right_metadata", "right_cases", "message"),
    [
        (
            {**METADATA, "dataset": {"path": "other", "versions": [], "splits": [], "n_cases": 1}},
            CRAG,
            "same dataset",
        ),
        ({**METADATA, "models": {"generation": "different"}}, CRAG, "same models"),
        ({**METADATA, "pricing": {"version": "different"}}, CRAG, "same token pricing"),
        ({**METADATA, "params": {**METADATA["params"], "top_k": 8}}, CRAG, "same top_k"),
        (METADATA, [{**CRAG[0], "case_id": "different"}], "case IDs"),
        (METADATA, [{**CRAG[0], "split": "holdout"}], "case splits"),
    ],
)
def test_incomparable_runs_are_rejected(right_metadata, right_cases, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compare_records(METADATA, right_metadata, BASELINE, right_cases)
