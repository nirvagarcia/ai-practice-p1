"""Reproducible run metadata and report persistence. Never stores keys or prompts."""

import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from bank_assistant.config.settings import Settings
from evaluation.common.datasets import EvalCase

RESULTS_DIR = Path("evaluation/results")
TRACKED_PACKAGES = (
    "langchain-core",
    "langchain-openai",
    "langgraph",
    "chromadb",
    "openai",
    "ragas",
)


def run_metadata(
    suite: str,
    variant: str,
    dataset: Path,
    cases: list[EvalCase],
    settings: Settings,
    params: dict[str, Any],
) -> dict[str, Any]:
    created_at = datetime.now(UTC)
    return {
        "run_id": f"{suite}-{variant}-{created_at:%Y%m%dT%H%M%SZ}",
        "created_at": created_at.isoformat(),
        "code": _git_state(),
        "packages": {name: _version(name) for name in TRACKED_PACKAGES},
        "dataset": {
            "path": dataset.as_posix(),
            "versions": sorted({case.dataset_version for case in cases}),
            "splits": sorted({case.split for case in cases}),
            "n_cases": len(cases),
        },
        "models": {
            "router": settings.router_model,
            "generation": settings.generation_model,
            "grader": settings.grader_model,
            "embedding": settings.embedding_model,
            "evaluator": settings.evaluator_model,
        },
        "pricing": {
            "version": settings.pricing_version,
            "input_usd_per_million_tokens": settings.input_cost_usd_per_million_tokens,
            "output_usd_per_million_tokens": settings.output_cost_usd_per_million_tokens,
        },
        "params": params,
    }


def write_report(
    report: dict[str, Any], results_dir: Path = RESULTS_DIR, output: Path | None = None
) -> Path:
    path = output or results_dir / f"{report['metadata']['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), "utf-8")
    return path


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
