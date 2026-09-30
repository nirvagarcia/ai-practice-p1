"""Pure metric functions shared by all evaluation suites."""

import math
from collections import Counter
from collections.abc import Collection, Sequence


def hit_at_k(relevant: Collection[str], retrieved: Sequence[str]) -> float:
    return float(any(item in relevant for item in retrieved))


def reciprocal_rank(relevant: Collection[str], retrieved: Sequence[str]) -> float:
    return next((1 / rank for rank, item in enumerate(retrieved, 1) if item in relevant), 0.0)


def recall_at_k(relevant: Collection[str], retrieved: Sequence[str]) -> float:
    return len(set(relevant) & set(retrieved)) / len(relevant)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile; report sample size alongside it, small samples are unstable."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(pct / 100 * len(ordered)) - 1, 0)]


def latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95)}


def classification_report(
    expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict:
    pairs = Counter(zip(expected, predicted, strict=True))
    per_class = {}
    for label in labels:
        tp = pairs[(label, label)]
        support = sum(n for (exp, _), n in pairs.items() if exp == label)
        predicted_as = sum(n for (_, pred), n in pairs.items() if pred == label)
        precision = tp / predicted_as if predicted_as else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    return {
        "accuracy": sum(pairs[(label, label)] for label in labels) / len(expected),
        "macro_f1": mean([metrics["f1"] for metrics in per_class.values()]),
        "per_class": per_class,
        "confusion": {
            exp: {p: pairs[(exp, p)] for p in sorted({*labels, *predicted}) if pairs[(exp, p)]}
            for exp in labels
        },
    }
