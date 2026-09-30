"""RAG evaluation: `python -m evaluation.rag.run --variant rag-v1 [--split dev] [--ragas]`."""

from collections.abc import Callable
from functools import partial
from typing import Any

from bank_assistant.application.bootstrap import build_faq
from bank_assistant.config.settings import ConfigurationError, Settings, get_settings
from bank_assistant.rag.chunking import CHUNKING_VERSION
from bank_assistant.rag.grader import GRADER_PROMPT_VERSION
from bank_assistant.rag.query_rewriter import REWRITER_PROMPT_VERSION
from bank_assistant.workflows.faq.baseline import FAQ_PROMPT_VERSION, BaselineFaq, FaqResult
from bank_assistant.workflows.faq.crag import CorrectiveFaq
from evaluation.common.datasets import EVAL_DIR, RagCase, load_cases
from evaluation.common.metrics import (
    hit_at_k,
    latency_summary,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.common.report import run_metadata
from evaluation.common.runner import Measured, eval_parser, measure, publish

FaqWorkflow = BaselineFaq | CorrectiveFaq
VARIANTS: dict[str, Callable[[Settings], FaqWorkflow]] = {
    "rag-v1": partial(build_faq, strategy="rag-v1"),
    "crag": partial(build_faq, strategy="crag"),
}
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def case_record(case: RagCase, run: Measured[FaqResult], settings: Settings) -> dict[str, Any]:
    record = {"case_id": case.case_id, "split": case.split, "answerable": case.answerable}
    record |= run.record()
    if run.value is None:
        return record
    result = run.value
    retrieved = [chunk.chunk_id for chunk in result.retrieval.chunks]
    record |= {
        "retrieved_ids": retrieved,
        "scores": [round(chunk.score, 4) for chunk in result.retrieval.chunks],
        "retrieval_ms": round(result.retrieval.duration_ms, 1),
        "grounded": result.grounded,
        "retrieval_attempts": result.retrieval_attempts,
        "source_ids": list(result.source_ids),
        "answer": result.answer,
    }
    if case.answerable:
        relevant = case.relevant_chunk_ids
        record |= {
            "hit": hit_at_k(relevant, retrieved),
            "reciprocal_rank": reciprocal_rank(relevant, retrieved),
            "recall": recall_at_k(relevant, retrieved),
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
    ok = [r for r in records if r["error"] is None]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    summary = {
        "n_cases": len(records),
        "errors": len(records) - len(ok),
        "retrieval": {
            "hit_rate": mean([r["hit"] for r in answerable]),
            "mrr": mean([r["reciprocal_rank"] for r in answerable]),
            "recall_at_k": mean([r["recall"] for r in answerable]),
        },
        "abstention": {
            "correct_on_unanswerable": mean([float(not r["grounded"]) for r in unanswerable]),
            "false_on_answerable": mean([float(not r["grounded"]) for r in answerable]),
        },
        "latency_ms": {
            "retrieval": latency_summary([r["retrieval_ms"] for r in ok]),
            "total": latency_summary([r["duration_ms"] for r in ok]),
        },
        "tokens_per_request": {
            "input": _mean_field(ok, "input_tokens"),
            "output": _mean_field(ok, "output_tokens"),
        },
        "estimated_cost_usd_per_request": _mean_field(ok, "estimated_cost_usd"),
        "retrieval_attempts_per_request": mean([r["retrieval_attempts"] for r in ok]),
    }
    scored = [r["ragas"] for r in records if "ragas" in r and "error" not in r["ragas"]]
    if scored:
        summary["ragas"] = {name: mean([s[name] for s in scored]) for name in RAGAS_METRICS}
        summary["ragas"]["n_scored"] = len(scored)
    return summary


def _mean_field(records: list[dict[str, Any]], field: str) -> float | None:
    return mean([record[field] for record in records if record.get(field) is not None])


def compare_records(
    baseline_metadata: dict[str, Any],
    corrective_metadata: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    corrective_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if baseline_metadata["dataset"] != corrective_metadata["dataset"]:
        raise ValueError("RAG variants must use the same dataset version, split and cases")
    if baseline_metadata["models"] != corrective_metadata["models"]:
        raise ValueError("RAG variants must use the same models")
    if baseline_metadata.get("pricing") != corrective_metadata.get("pricing"):
        raise ValueError("RAG variants must use the same token pricing")
    for key in ("top_k", "chunking_version", "generation_prompt"):
        if baseline_metadata["params"].get(key) != corrective_metadata["params"].get(key):
            raise ValueError(f"RAG variants must use the same {key}")
    baseline = {record["case_id"]: record for record in baseline_records}
    corrective = {record["case_id"]: record for record in corrective_records}
    if (
        baseline.keys() != corrective.keys()
        or len(baseline) != len(baseline_records)
        or len(corrective) != len(corrective_records)
    ):
        raise ValueError("RAG variants must contain identical case IDs")
    if any(baseline[key].get("split") != corrective[key].get("split") for key in baseline):
        raise ValueError("RAG variants must use identical case splits")
    paired = []
    for case_id, first in baseline.items():
        second = corrective[case_id]
        paired.append(
            {
                "case_id": case_id,
                "answer_changed": first.get("answer") != second.get("answer"),
                "grounded_changed": first.get("grounded") != second.get("grounded"),
                "retrieval_changed": first.get("retrieved_ids") != second.get("retrieved_ids"),
                "baseline_latency_ms": first.get("duration_ms"),
                "crag_latency_ms": second.get("duration_ms"),
                "latency_delta_ms": _delta(first.get("duration_ms"), second.get("duration_ms")),
            }
        )
    return {
        "variants": ["rag-v1", "crag"],
        "paired_cases": paired,
        "changed_answers": sum(case["answer_changed"] for case in paired),
        "changed_retrievals": sum(case["retrieval_changed"] for case in paired),
        "recommendation": (
            "CRAG is experimental; retain it only if paired quality gains justify cost and latency."
        ),
    }


def _delta(first: Any, second: Any) -> float | None:
    return second - first if first is not None and second is not None else None


def add_ragas_scores(
    records: list[dict[str, Any]],
    cases: list[RagCase],
    runs: list[Measured[FaqResult]],
    settings: Settings,
) -> None:
    from evaluation.rag.ragas_adapter import RagasSample, RagasScorer  # optional `eval` extra

    targets = [
        (
            record,
            RagasSample(
                case.question,
                run.value.answer,
                [c.text for c in run.value.retrieval.chunks],
                case.reference,
            ),
        )
        for record, case, run in zip(records, cases, runs, strict=True)
        if case.answerable and run.value is not None and run.value.retrieval.chunks
    ]
    scores = RagasScorer(settings).score_all([sample for _, sample in targets])
    for (record, _), score in zip(targets, scores, strict=True):
        record["ragas"] = score


def main(argv: list[str] | None = None) -> None:
    parser = eval_parser("Evaluate a RAG variant.", EVAL_DIR / "rag.jsonl")
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--variant", choices=list(VARIANTS))
    choices.add_argument("--compare", help="paired variants, e.g. rag-v1,crag")
    parser.add_argument("--ragas", action="store_true", help="add LLM-judged RAGAS metrics")
    args = parser.parse_args(argv)

    settings = get_settings()
    cases = load_cases(args.dataset, RagCase, args.split)
    variants = (
        [item.strip() for item in args.compare.split(",")]
        if args.compare
        else [args.variant or "rag-v1"]
    )
    if len(set(variants)) != len(variants) or any(name not in VARIANTS for name in variants):
        parser.error("--compare requires distinct known variants")
    if args.compare and variants != ["rag-v1", "crag"]:
        parser.error("--compare currently requires the ordered pair rag-v1,crag")

    try:
        results = {}
        for variant in variants:
            faq = VARIANTS[variant](settings)
            runs = [measure(lambda case=case, faq=faq: faq.answer(case.question)) for case in cases]
            records = [
                case_record(case, run, settings) for case, run in zip(cases, runs, strict=True)
            ]
            if args.ragas:
                add_ragas_scores(records, cases, runs, settings)
            metadata = run_metadata(
                "rag",
                variant,
                args.dataset,
                cases,
                settings,
                {
                    "top_k": settings.retrieval_top_k,
                    "chunking_version": CHUNKING_VERSION,
                    "generation_prompt": FAQ_PROMPT_VERSION,
                    "max_retrieval_attempts": (
                        settings.max_retrieval_attempts if variant == "crag" else 1
                    ),
                    "grader_prompt": GRADER_PROMPT_VERSION if variant == "crag" else None,
                    "rewriter_prompt": REWRITER_PROMPT_VERSION if variant == "crag" else None,
                },
            )
            results[variant] = {
                "metadata": metadata,
                "summary": summarize(records),
                "cases": records,
            }
    except ConfigurationError as error:
        parser.exit(2, f"{error}\n")
    if args.compare:
        left, right = variants
        comparison = compare_records(
            results[left]["metadata"],
            results[right]["metadata"],
            results[left]["cases"],
            results[right]["cases"],
        )
        report = {
            "metadata": run_metadata(
                "rag",
                f"{left}-vs-{right}",
                args.dataset,
                cases,
                settings,
                {
                    "top_k": settings.retrieval_top_k,
                    "chunking_version": CHUNKING_VERSION,
                    "ragas": args.ragas,
                },
            ),
            "variants": results,
            "summary": {name: result["summary"] for name, result in results.items()},
            "comparison": comparison,
        }
    else:
        report = results[variants[0]]
    publish(report)


if __name__ == "__main__":
    main()
