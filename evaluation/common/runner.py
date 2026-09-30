"""Shared runner plumbing: CLI arguments, timing/token capture and report output."""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback

from evaluation.common.report import write_report


@dataclass(frozen=True)
class Measured[T]:
    value: T | None
    error: str | None
    duration_ms: float
    input_tokens: int | None
    output_tokens: int | None

    def record(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def measure[T](call: Callable[[], T]) -> Measured[T]:
    """Run one case; failures are recorded by type only, so no content leaks into reports."""
    started = perf_counter()
    with get_usage_metadata_callback() as usage:
        try:
            value, error = call(), None
        except Exception as exc:
            value, error = None, type(exc).__name__
    totals = usage.usage_metadata.values()
    totals = list(totals)
    return Measured(
        value=value,
        error=error,
        duration_ms=(perf_counter() - started) * 1000,
        input_tokens=sum(u.get("input_tokens", 0) for u in totals) if totals else None,
        output_tokens=sum(u.get("output_tokens", 0) for u in totals) if totals else None,
    )


def eval_parser(description: str, default_dataset: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--split", choices=["dev", "holdout"], help="default: all cases")
    return parser


def publish(report: dict[str, Any], output: Path | None = None) -> None:
    path = write_report(report, output=output)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Report written to {path}")
