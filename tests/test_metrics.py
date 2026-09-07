"""Tests for :mod:`llm_forensic_timeline.metrics`."""

import sys
from types import SimpleNamespace

import pytest

from llm_forensic_timeline import metrics
from llm_forensic_timeline.metrics import (
    BERTSCORE_FIELDS,
    BertScoreMetric,
    BleuMetric,
    MetricSuite,
    RougeMetric,
    build_default_suite,
    drop_empty_pairs,
    select_device,
)

from conftest import FakeScorer


class FakeBackend:
    """Stands in for a Hugging Face ``evaluate`` metric."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def compute(self, predictions, references):
        self.calls.append((predictions, references))
        return self.results


def test_drop_empty_pairs_keeps_aligned_pairs():
    predictions = ["a shutdown event", "   ", "a google search"]
    references = ["a shutdown event", "a web visit", ""]

    kept_predictions, kept_references = drop_empty_pairs(predictions, references)

    assert kept_predictions == ["a shutdown event"]
    assert kept_references == ["a shutdown event"]


def test_drop_empty_pairs_returns_empty_lists_when_nothing_survives():
    assert drop_empty_pairs(["", " "], ["", " "]) == ([], [])


def fake_torch(cuda_available, mps_available):
    """Build a stand-in for the ``torch`` module."""
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        ),
    )


@pytest.mark.parametrize(
    "cuda_available, mps_available, expected",
    [
        (True, False, "cuda"),
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_select_device_prefers_gpu(monkeypatch, cuda_available, mps_available, expected):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available, mps_available))

    assert select_device() == expected


def test_bleu_metric_selects_only_its_own_fields():
    backend = FakeBackend(
        {
            "bleu": 0.83,
            "precisions": [0.9],
            "brevity_penalty": 1.0,
            "length_ratio": 1.02,
            "translation_length": 72,
            "reference_length": 70,
        }
    )
    metric = BleuMetric(backend=backend)

    scores = metric.compute(["a"], ["b"])

    assert set(scores) == set(metric.fields)
    assert scores["bleu"] == 0.83
    assert "precisions" not in scores


def test_rouge_metric_casts_to_float():
    backend = FakeBackend(
        {"rouge1": 0.79, "rouge2": 0.64, "rougeL": 0.79, "rougeLsum": 0.79}
    )
    metric = RougeMetric(backend=backend)

    scores = metric.compute(["a"], ["b"])

    assert scores == {
        "rouge1": 0.79,
        "rouge2": 0.64,
        "rougeL": 0.79,
        "rougeLsum": 0.79,
    }
    assert all(isinstance(value, float) for value in scores.values())


def test_bertscore_metric_averages_the_per_pair_triples():
    scorer = FakeScorer(scores=[(1.0, 0.5, 0.6), (0.0, 0.5, 0.4)])
    metric = BertScoreMetric(scorer=scorer)

    scores = metric.compute(["a", "b"], ["c", "d"])

    assert scores == {
        "bertscore_precision": 0.5,
        "bertscore_recall": 0.5,
        "bertscore_f1": pytest.approx(0.5),
    }


def test_bertscore_metric_skips_blank_pairs_before_scoring():
    scorer = FakeScorer()
    metric = BertScoreMetric(scorer=scorer)

    metric.compute(["a shutdown", ""], ["a shutdown", "a web visit"])

    assert scorer.calls == [(["a shutdown"], ["a shutdown"])]


def test_bertscore_metric_returns_zeros_when_every_pair_is_blank():
    scorer = FakeScorer()
    metric = BertScoreMetric(scorer=scorer)

    scores = metric.compute(["", " "], ["", " "])

    assert scores == dict.fromkeys(BERTSCORE_FIELDS, 0)
    assert scorer.calls == []


def test_metric_zero_scores_covers_every_field():
    assert BleuMetric().zero_scores() == dict.fromkeys(BleuMetric.fields, 0)


def test_metric_suite_merges_fields_in_order(fake_metric):
    suite = MetricSuite([fake_metric, BertScoreMetric(scorer=FakeScorer())])

    assert suite.fields == ["alpha", "beta", *BERTSCORE_FIELDS]

    scores = suite.compute(["a"], ["b"])
    assert set(scores) == set(suite.fields)


def test_metric_suite_zero_scores_covers_every_field(fake_suite):
    assert fake_suite.zero_scores() == {"alpha": 0, "beta": 0}


def test_build_default_suite_can_drop_bertscore():
    suite = build_default_suite(with_bertscore=False)

    assert [type(metric) for metric in suite.metrics] == [BleuMetric, RougeMetric]
    assert not any(field.startswith("bertscore") for field in suite.fields)


def test_build_default_suite_uses_the_modernbert_large_checkpoint():
    suite = build_default_suite()

    bertscore = suite.metrics[-1]
    assert isinstance(bertscore, BertScoreMetric)
    assert bertscore.model_id == metrics.DEFAULT_BERTSCORE_MODEL
    assert "ModernBERT-large" in bertscore.model_id
