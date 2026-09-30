"""End-to-end evaluation: `python -m evaluation.end_to_end.run --smoke --report path.json`."""

from pathlib import Path
from typing import Any

from bank_assistant.application.bootstrap import build_assistant
from bank_assistant.config.settings import ConfigurationError, Settings, get_settings
from bank_assistant.domain.schemas import Principal
from bank_assistant.state.schemas import Outcome
from evaluation.common.datasets import EVAL_DIR, EndToEndCase, load_cases
from evaluation.common.metrics import classification_report, latency_summary, mean
from evaluation.common.report import run_metadata
from evaluation.common.runner import Measured, eval_parser, measure, publish


def case_record(case: EndToEndCase, run: Measured, settings: Settings) -> dict[str, Any]:
    result = run.value
    record = {
        "case_id": case.case_id,
        "split": case.split,
        "expected_intent": case.expected_intent,
        "expected_outcome": case.expected_outcome,
        **run.record(),
    }
    if result is None:
        return record
    record |= {
        "predicted_intent": result.intent,
        "predicted_outcome": result.outcome,
        "intent_correct": result.intent == case.expected_intent,
        "outcome_correct": result.outcome == case.expected_outcome,
        "completed": result.outcome not in {Outcome.FAILED, Outcome.REJECTED, Outcome.BLOCKED},
    }
    if (
        run.input_tokens is not None
        and run.output_tokens is not None
        and settings.input_cost_usd_per_million_tokens is not None
        and settings.output_cost_usd_per_million_tokens is not None
    ):
        record["estimated_cost_usd"] = (
            run.input_tokens * settings.input_cost_usd_per_million_tokens
            + run.output_tokens * settings.output_cost_usd_per_million_tokens
        ) / 1_000_000
    else:
        record["estimated_cost_usd"] = None
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if "predicted_outcome" in record]
    expected = [str(record["expected_intent"]) for record in records]
    predicted = [str(record.get("predicted_intent") or "unknown") for record in records]
    labels = sorted(set(expected) | set(predicted))
    return {
        "n_cases": len(records),
        "runtime_errors": sum(record["error"] is not None for record in records),
        "intent": classification_report(expected, predicted, labels),
        "outcome_accuracy": mean([float(r["outcome_correct"]) for r in valid]),
        "completion_rate": mean([float(r["completed"]) for r in valid]),
        "latency_ms": latency_summary([record["duration_ms"] for record in records]),
        "tokens_per_request": {
            "input": _mean_field(records, "input_tokens"),
            "output": _mean_field(records, "output_tokens"),
        },
        "estimated_cost_usd_per_request": _mean_field(records, "estimated_cost_usd"),
    }


def _mean_field(records: list[dict[str, Any]], field: str) -> float | None:
    return mean([r[field] for r in records if r.get(field) is not None])


def main(argv: list[str] | None = None) -> None:
    parser = eval_parser("Evaluate end-to-end assistant objectives.", EVAL_DIR / "end_to_end.jsonl")
    parser.add_argument("--smoke", action="store_true", help="run up to five dev cases")
    parser.add_argument("--report", type=Path, help="optional explicit JSON report path")
    args = parser.parse_args(argv)
    settings = get_settings()
    cases = load_cases(args.dataset, EndToEndCase, args.split or ("dev" if args.smoke else None))
    if args.smoke:
        cases = cases[:5]
    try:
        assistant = build_assistant(settings)
    except ConfigurationError as error:
        parser.exit(2, f"{error}\n")
    runs = [
        measure(
            lambda case=case: assistant.ask(case.message, Principal(customer_id=case.customer_id))
        )
        for case in cases
    ]
    records = [case_record(case, run, settings) for case, run in zip(cases, runs, strict=True)]
    report = {
        "metadata": run_metadata(
            "end-to-end",
            "root-graph",
            args.dataset,
            cases,
            settings,
            {
                "max_graph_steps": settings.max_graph_steps,
                "max_total_tokens_per_request": settings.max_total_tokens_per_request,
                "max_tokens_per_call": settings.max_tokens_per_request,
                "rag_strategy": settings.rag_strategy,
            },
        ),
        "summary": summarize(records),
        "cases": records,
    }
    publish(report, args.report)


if __name__ == "__main__":
    main()
