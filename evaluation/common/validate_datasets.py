"""Dataset integrity checks: `python -m evaluation.common.validate_datasets`."""

import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bank_assistant.domain.schemas import Intent
from bank_assistant.rag.ingest import load_chunks
from evaluation.common.datasets import (
    EVAL_DIR,
    EndToEndCase,
    EvalCase,
    RagCase,
    RoutingCase,
    ToolCase,
    load_cases,
)

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")


@dataclass(frozen=True)
class DatasetSpec:
    model: type[EvalCase]
    label: Callable[[Any], str]
    text: Callable[[Any], str]


SPECS = {
    "rag.jsonl": DatasetSpec(
        RagCase, lambda c: "answerable" if c.answerable else "unanswerable", lambda c: c.question
    ),
    "routing.jsonl": DatasetSpec(
        RoutingCase, lambda c: str(c.expected_intent), lambda c: c.message
    ),
    "tools.jsonl": DatasetSpec(ToolCase, lambda c: str(c.expected_tool), lambda c: c.message),
    "end_to_end.jsonl": DatasetSpec(
        EndToEndCase, lambda c: f"{c.expected_intent}/{c.expected_outcome}", lambda c: c.message
    ),
}


def validate(
    eval_dir: Path = EVAL_DIR, knowledge_base: Path = KNOWLEDGE_BASE_DIR
) -> tuple[list[str], dict[str, Counter]]:
    errors: list[str] = []
    summary: dict[str, Counter] = {}
    for name, spec in SPECS.items():
        try:
            cases = load_cases(eval_dir / name, spec.model)
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue
        errors += [f"{name}: {problem}" for problem in _generic_problems(cases, spec)]
        summary[name] = Counter(spec.label(case) for case in cases)

        if spec.model is RagCase:
            known = {chunk.chunk_id for chunk in load_chunks(knowledge_base)}
            errors += [
                f"{name}: {case.case_id} references unknown chunk '{chunk_id}'"
                for case in cases
                for chunk_id in case.relevant_chunk_ids
                if chunk_id not in known
            ]
            if all(case.answerable for case in cases):
                errors.append(f"{name}: needs at least one unanswerable case")
        if spec.model is RoutingCase:
            if missing := set(Intent) - {case.expected_intent for case in cases}:
                errors.append(f"{name}: no cases for intents {sorted(missing)}")
            if not any("adversarial" in case.tags for case in cases):
                errors.append(f"{name}: needs at least one adversarial case")
    return errors, summary


def _generic_problems(cases: list[EvalCase], spec: DatasetSpec) -> list[str]:
    problems = []
    if not cases:
        return ["dataset is empty"]
    if duplicated := _duplicates(case.case_id for case in cases):
        problems.append(f"duplicate case_id {duplicated}")
    if duplicated := _duplicates(spec.text(case).casefold() for case in cases):
        problems.append(f"duplicate inputs {duplicated}")
    if len(versions := {case.dataset_version for case in cases}) > 1:
        problems.append(f"mixed dataset_version {sorted(versions)}")
    if {case.split for case in cases} != {"dev", "holdout"}:
        problems.append("needs both dev and holdout splits")
    return problems


def _duplicates(values: Any) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def main() -> None:
    errors, summary = validate()
    for name, counts in summary.items():
        print(f"{name}: {sum(counts.values())} cases {dict(sorted(counts.items()))}")
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
