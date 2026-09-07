"""Standardized evaluation of LLM-based digital forensic timeline analysis.

The package implements the methodology of the DFRWS APAC 2025 paper *Towards a
standardized methodology and dataset for evaluating LLM-based digital forensic
timeline analysis*. It drives an LLM over a Plaso timeline and scores the
answers against a ground truth with BLEU, ROUGE, and BERTScore.
"""

from .evaluation import (
    evaluate_directory,
    evaluate_file,
    load_pair,
    read_json_file,
    read_txt_file,
    write_results_csv,
)
from .metrics import (
    DEFAULT_BERTSCORE_MODEL,
    BertScoreMetric,
    BleuMetric,
    MetricSuite,
    RougeMetric,
    build_default_suite,
    select_device,
)

__version__ = "0.0.2"

__all__ = [
    "BertScoreMetric",
    "BleuMetric",
    "DEFAULT_BERTSCORE_MODEL",
    "MetricSuite",
    "RougeMetric",
    "build_default_suite",
    "evaluate_directory",
    "evaluate_file",
    "load_pair",
    "read_json_file",
    "read_txt_file",
    "select_device",
    "write_results_csv",
    "__version__",
]
