"""Routing evaluation: `python -m evaluation.routing.run [--split dev]`."""

from typing import Any

from bank_assistant.application.bootstrap import build_triage
from bank_assistant.config.settings import ConfigurationError, get_settings
from bank_assistant.domain.schemas import Intent
from bank_assistant.workflows.triage import TriageDecision
from evaluation.common.datasets import EVAL_DIR, RoutingCase, load_cases
from evaluation.common.metrics import classification_report, latency_summary, mean
from evaluation.common.report import run_metadata
from evaluation.common.runner import Measured, eval_parser, measure, publish

PARSE_FAILURE = "invalid"


def case_record(case: RoutingCase, run: Measured[TriageDecision]) -> dict[str, Any]:
    decision = run.value
    return {
        "case_id": case.case_id,
        "split": case.split,
        "tags": list(case.tags),
        "expected": case.expected_intent,
        "predicted": decision.intent if decision else PARSE_FAILURE,
        "confidence": decision.confidence if decision else None,
        **run.record(),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = list(Intent)
    report = classification_report(
        [r["expected"] for r in records], [r["predicted"] for r in records], labels
    )
    return {
        "n_cases": len(records),
        "errors": sum(r["error"] is not None for r in records),
        **report,
        # Missed card blocks are the costliest routing error: list them explicitly.
        "card_block_false_negatives": [
            r["case_id"]
            for r in records
            if r["expected"] == Intent.CARD_BLOCK and r["predicted"] != Intent.CARD_BLOCK
        ],
        "latency_ms": latency_summary([r["duration_ms"] for r in records]),
        "tokens_per_request": {
            "input": mean([r["input_tokens"] for r in records if r["input_tokens"] is not None]),
            "output": mean([r["output_tokens"] for r in records if r["output_tokens"] is not None]),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = eval_parser("Evaluate the triage router.", EVAL_DIR / "routing.jsonl")
    args = parser.parse_args(argv)

    settings = get_settings()
    cases = load_cases(args.dataset, RoutingCase, args.split)
    try:
        triage = build_triage(settings)
    except ConfigurationError as error:
        parser.exit(2, f"{error}\n")

    runs = [measure(lambda case=case: triage.classify(case.message)) for case in cases]
    records = [case_record(case, run) for case, run in zip(cases, runs, strict=True)]
    params = {"min_confidence": settings.router_min_confidence}
    publish(
        {
            "metadata": run_metadata(
                "routing", "llm-router", args.dataset, cases, settings, params
            ),
            "summary": summarize(records),
            "cases": records,
        }
    )


if __name__ == "__main__":
    main()
