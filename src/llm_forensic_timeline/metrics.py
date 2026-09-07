"""Evaluation metrics for LLM-based forensic timeline analysis.

The module wraps the three metrics discussed in the paper: BLEU and ROUGE from
the Hugging Face ``evaluate`` library, and BERTScore from ``modern-bert-score``.

BLEU and ROUGE only measure lexical overlap, so an answer that is worded
differently but forensically correct scores poorly. BERTScore is semantic-aware
and complements them, as recommended in Sec. 4.4.2 of the paper.

Every backend is loaded lazily. Importing this module therefore costs nothing,
and the unit tests can inject their own scorer instead of downloading a model.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

#: The strongest checkpoint shipped with ``modern-bert-score``. It is the
#: largest one, and its 8192-token context keeps the long JSON entries of a
#: forensic timeline intact, while the RoBERTa checkpoints truncate at 512.
DEFAULT_BERTSCORE_MODEL = "LazerLambda/ModernBERT-large-ModBERTScore-19"

BLEU_FIELDS = (
    "bleu",
    "brevity_penalty",
    "length_ratio",
    "translation_length",
    "reference_length",
)
ROUGE_FIELDS = ("rouge1", "rouge2", "rougeL", "rougeLsum")
BERTSCORE_FIELDS = ("bertscore_precision", "bertscore_recall", "bertscore_f1")


def select_device() -> str:
    """Return the best device available for the BERTScore model.

    :return: ``"cuda"``, ``"mps"``, or ``"cpu"``.
    :rtype: str
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def drop_empty_pairs(
    predictions: Sequence[str], references: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """Drop the pairs where either side is blank.

    An empty string carries no token to match, which makes BERTScore fail on an
    empty similarity matrix.

    :param predictions: The LLM answers.
    :param references: The ground truth entries.
    :return: The surviving predictions and references, still aligned.
    :rtype: tuple[list[str], list[str]]
    """
    kept = [
        (prediction, reference)
        for prediction, reference in zip(predictions, references)
        if prediction.strip() and reference.strip()
    ]
    if not kept:
        return [], []

    surviving_predictions, surviving_references = zip(*kept)
    return list(surviving_predictions), list(surviving_references)


class Metric:
    """Base class for a metric that scores predictions against references."""

    #: The CSV column names this metric contributes, in order.
    fields: Tuple[str, ...] = ()

    def compute(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> Dict[str, float]:
        """Score ``predictions`` against ``references``.

        :param predictions: The LLM answers.
        :param references: The ground truth entries.
        :return: One value per name in :attr:`fields`.
        :rtype: dict[str, float]
        """
        raise NotImplementedError

    def zero_scores(self) -> Dict[str, float]:
        """Return a zeroed result, used when there is nothing to score.

        :return: Every field of the metric set to ``0``.
        :rtype: dict[str, float]
        """
        return {field: 0 for field in self.fields}


class BleuMetric(Metric):
    """BLEU through the Hugging Face ``evaluate`` library."""

    fields = BLEU_FIELDS

    def __init__(self, backend: Any = None):
        """
        :param backend: A preloaded ``evaluate`` metric. Loaded on first use
            when omitted, which is what the tests replace.
        """
        self._backend = backend

    @property
    def backend(self) -> Any:
        """The underlying ``evaluate`` metric, loaded on first access."""
        if self._backend is None:
            import evaluate

            self._backend = evaluate.load("bleu")
        return self._backend

    def compute(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> Dict[str, float]:
        results = self.backend.compute(
            predictions=list(predictions), references=list(references)
        )
        return {field: results[field] for field in self.fields}


class RougeMetric(Metric):
    """ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum through ``evaluate``."""

    fields = ROUGE_FIELDS

    def __init__(self, backend: Any = None):
        """
        :param backend: A preloaded ``evaluate`` metric. Loaded on first use
            when omitted.
        """
        self._backend = backend

    @property
    def backend(self) -> Any:
        """The underlying ``evaluate`` metric, loaded on first access."""
        if self._backend is None:
            import evaluate

            self._backend = evaluate.load("rouge")
        return self._backend

    def compute(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> Dict[str, float]:
        results = self.backend.compute(
            predictions=list(predictions), references=list(references)
        )
        return {field: float(results[field]) for field in self.fields}


class BertScoreMetric(Metric):
    """BERTScore through ``modern-bert-score``.

    The scorer returns one ``(precision, recall, f1)`` triple per pair. The
    triples are averaged so that a file yields a single score, the same way
    BLEU and ROUGE are aggregated.
    """

    fields = BERTSCORE_FIELDS

    def __init__(
        self,
        model_id: str = DEFAULT_BERTSCORE_MODEL,
        device: str = None,
        baseline_rescaling: bool = True,
        scorer: Any = None,
    ):
        """
        :param model_id: The Hugging Face checkpoint to embed with.
        :param device: ``"cuda"``, ``"mps"``, or ``"cpu"``. Detected when
            omitted.
        :param baseline_rescaling: Spread the scores over a readable range
            instead of leaving them compressed near 0.8, as recommended by the
            BERTScore authors. Rescaled scores can be slightly negative.
        :param scorer: A preloaded callable. Loaded on first use when omitted,
            which is what the tests replace.
        """
        self.model_id = model_id
        self.device = device
        self.baseline_rescaling = baseline_rescaling
        self._scorer = scorer

    @property
    def scorer(self) -> Any:
        """The underlying scorer, loaded on first access.

        Loading downloads the model weights (about 1.6 GB for the default
        checkpoint) the first time and caches them afterwards.
        """
        if self._scorer is None:
            from modern_bert_score import BertScore

            device = self.device or select_device()
            self._scorer = BertScore(
                model_id=self.model_id,
                device=device,
                baseline_rescaling=self.baseline_rescaling,
            )
        return self._scorer

    def compute(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> Dict[str, float]:
        predictions, references = drop_empty_pairs(predictions, references)
        if not predictions:
            return self.zero_scores()

        scores = self.scorer(predictions, references)
        if not scores:
            return self.zero_scores()

        return {
            field: sum(score[index] for score in scores) / len(scores)
            for index, field in enumerate(self.fields)
        }


class MetricSuite:
    """A group of metrics scored together and written to one CSV row."""

    def __init__(self, metrics: Iterable[Metric]):
        """
        :param metrics: The metrics to run, in the order their columns appear.
        """
        self.metrics = list(metrics)

    @property
    def fields(self) -> List[str]:
        """Every column name contributed by the metrics, in order."""
        return [field for metric in self.metrics for field in metric.fields]

    def compute(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> Dict[str, float]:
        """Score ``predictions`` with every metric of the suite.

        :param predictions: The LLM answers.
        :param references: The ground truth entries.
        :return: The merged results of all metrics.
        :rtype: dict[str, float]
        """
        results: Dict[str, float] = {}
        for metric in self.metrics:
            results.update(metric.compute(predictions, references))
        return results

    def zero_scores(self) -> Dict[str, float]:
        """Return a zeroed result for every metric of the suite.

        :return: Every column of the suite set to ``0``.
        :rtype: dict[str, float]
        """
        results: Dict[str, float] = {}
        for metric in self.metrics:
            results.update(metric.zero_scores())
        return results


def build_default_suite(
    with_bertscore: bool = True,
    bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
    device: str = None,
) -> MetricSuite:
    """Build the suite used in the paper: BLEU, ROUGE, and BERTScore.

    :param with_bertscore: Drop BERTScore when ``False``, which avoids loading
        a transformer model.
    :param bertscore_model: The BERTScore checkpoint to embed with.
    :param device: The device for BERTScore. Detected when omitted.
    :return: The assembled suite.
    :rtype: MetricSuite
    """
    metrics: List[Metric] = [BleuMetric(), RougeMetric()]
    if with_bertscore:
        metrics.append(BertScoreMetric(model_id=bertscore_model, device=device))
    return MetricSuite(metrics)
