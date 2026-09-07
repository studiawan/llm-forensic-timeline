"""Shared fixtures for the test suite."""

import json

import pytest

from llm_forensic_timeline.metrics import Metric, MetricSuite


class FakeMetric(Metric):
    """A metric that returns fixed values without loading a model."""

    fields = ("alpha", "beta")

    def __init__(self):
        self.calls = []

    def compute(self, predictions, references):
        self.calls.append((list(predictions), list(references)))
        return {"alpha": float(len(predictions)), "beta": 0.5}


class FakeScorer:
    """Stands in for a ``modern_bert_score.BertScore`` instance."""

    def __init__(self, scores=None):
        self.scores = scores
        self.calls = []

    def __call__(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        if self.scores is not None:
            return self.scores
        return [(1.0, 1.0, 1.0) for _ in candidates]


@pytest.fixture
def fake_metric():
    """A metric that records its calls and returns fixed values."""
    return FakeMetric()


@pytest.fixture
def fake_suite(fake_metric):
    """A suite built around :class:`FakeMetric`."""
    return MetricSuite([fake_metric])


@pytest.fixture
def timeline_dirs(tmp_path):
    """A ground truth directory and a prediction directory holding one pair."""
    groundtruth_dir = tmp_path / "ground-truth"
    llm_dir = tmp_path / "chatgpt-with-knowledge"
    groundtruth_dir.mkdir()
    llm_dir.mkdir()

    event = {"id": 1002, "type": "Shutdown time", "description": "Windows shut down"}
    (groundtruth_dir / "14-shutdown.json").write_text(json.dumps([event]))
    (llm_dir / "14-shutdown.json").write_text(json.dumps([event]))

    return groundtruth_dir, llm_dir
