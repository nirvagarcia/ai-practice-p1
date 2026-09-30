import pytest

from evaluation.common.metrics import (
    classification_report,
    hit_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics() -> None:
    retrieved = ["a#0", "b#1", "c#2"]

    assert hit_at_k({"b#1"}, retrieved) == 1.0
    assert hit_at_k({"z#0"}, retrieved) == 0.0
    assert reciprocal_rank({"b#1", "c#2"}, retrieved) == pytest.approx(0.5)
    assert reciprocal_rank({"z#0"}, retrieved) == 0.0
    assert recall_at_k({"a#0", "z#0"}, retrieved) == pytest.approx(0.5)


def test_percentile_uses_nearest_rank() -> None:
    values = [float(v) for v in range(1, 21)]

    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0
    assert percentile([], 95) is None


def test_classification_report() -> None:
    report = classification_report(
        expected=["faq", "faq", "account", "card_block"],
        predicted=["faq", "account", "account", "invalid"],
        labels=["faq", "account", "card_block"],
    )

    assert report["accuracy"] == pytest.approx(0.5)
    assert report["per_class"]["faq"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }
    assert report["per_class"]["card_block"]["recall"] == 0.0
    assert report["confusion"]["card_block"] == {"invalid": 1}
