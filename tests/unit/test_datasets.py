from pathlib import Path

from evaluation.common.datasets import RagCase, load_cases
from evaluation.common.validate_datasets import SPECS, validate


def test_versioned_datasets_are_valid(eval_dir: Path, knowledge_base_dir: Path) -> None:
    errors, summary = validate(eval_dir, knowledge_base_dir)

    assert errors == []
    assert set(summary) == set(SPECS)


def test_validation_reports_unknown_chunks_and_coverage_gaps(
    tmp_path: Path, eval_dir: Path, knowledge_base_dir: Path
) -> None:
    for name in SPECS:
        (tmp_path / name).write_bytes((eval_dir / name).read_bytes())
    (tmp_path / "rag.jsonl").write_text(
        '{"case_id": "r1", "dataset_version": "v", "split": "dev", "question": "q",'
        ' "reference": "r", "relevant_chunk_ids": ["ghost#0"]}\n',
        "utf-8",
    )

    errors, _ = validate(tmp_path, knowledge_base_dir)

    assert any("unknown chunk 'ghost#0'" in error for error in errors)
    assert any("unanswerable" in error for error in errors)
    assert any("holdout" in error for error in errors)


def test_malformed_line_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "rag.jsonl"
    path.write_text('\n{"case_id": "r1"}\n', "utf-8")

    errors, _ = validate(tmp_path, tmp_path)

    assert any(f"{path}:2:" in error for error in errors)


def test_split_filter(eval_dir: Path) -> None:
    holdout = load_cases(eval_dir / "rag.jsonl", RagCase, split="holdout")

    assert holdout
    assert {case.split for case in holdout} == {"holdout"}
