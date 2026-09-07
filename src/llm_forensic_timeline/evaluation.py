"""Comparison of LLM answers against the ground truth.

The evaluation walks a ground truth directory, pairs every entry with the file
of the same name in the prediction directory, scores the pair with a
:class:`~llm_forensic_timeline.metrics.MetricSuite`, and writes one CSV row per
file.

Both JSON and plain text ground truth are supported, matching the tasks of the
paper: event summarization and rule-based anomaly detection produce JSON, while
running grep for specific terms produces plain text.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .metrics import MetricSuite

#: The ground truth extensions the evaluation knows how to read.
SUPPORTED_SUFFIXES = (".json", ".txt")


def read_json_file(file_path: str) -> Optional[object]:
    """Read a JSON file.

    :param file_path: Path to the file.
    :return: The decoded document, or ``None`` when the file does not exist.
    """
    try:
        with open(file_path, "r") as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def read_txt_file(file_path: str) -> Optional[str]:
    """Read a plain text file.

    :param file_path: Path to the file.
    :return: The file contents, or ``None`` when the file does not exist.
    """
    try:
        with open(file_path, "r") as handle:
            return handle.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def as_entry_list(data: object) -> List[object]:
    """Normalise a decoded JSON document into a list of entries.

    A single high-level event is stored as an object rather than an array, so
    it is wrapped to keep the rest of the pipeline uniform.

    :param data: The decoded JSON document.
    :return: The entries of the document.
    :rtype: list
    """
    if isinstance(data, dict):
        return [data]
    return list(data)


def truncate_to_common_length(
    predictions: Sequence, references: Sequence
) -> Tuple[List, List]:
    """Cut both sequences down to the length of the shorter one.

    The metrics need aligned pairs, and an LLM regularly returns more or fewer
    entries than the ground truth holds.

    :param predictions: The LLM entries.
    :param references: The ground truth entries.
    :return: The truncated predictions and references.
    :rtype: tuple[list, list]
    """
    common_length = min(len(predictions), len(references))
    return list(predictions[:common_length]), list(references[:common_length])


def load_json_pair(
    groundtruth_path: str, llm_path: str
) -> Optional[Tuple[List[str], List[str]]]:
    """Load a JSON ground truth file and its prediction as aligned strings.

    Each entry is serialised back to JSON so that the metrics compare the
    structure and the values, not just the free text.

    :param groundtruth_path: Path to the ground truth file.
    :param llm_path: Path to the LLM answer.
    :return: The aligned predictions and references, or ``None`` when the
        prediction is missing.
    """
    groundtruth_data = read_json_file(groundtruth_path)
    llm_data = read_json_file(llm_path)
    if groundtruth_data is None or llm_data is None:
        return None

    references = as_entry_list(groundtruth_data)
    predictions = as_entry_list(llm_data)
    predictions, references = truncate_to_common_length(predictions, references)

    return (
        [json.dumps(entry) for entry in predictions],
        [json.dumps(entry) for entry in references],
    )


def load_txt_pair(
    groundtruth_path: str, llm_path: str
) -> Optional[Tuple[List[str], List[str]]]:
    """Load a plain text ground truth file and its prediction, line by line.

    :param groundtruth_path: Path to the ground truth file.
    :param llm_path: Path to the LLM answer.
    :return: The aligned predictions and references, or ``None`` when the
        prediction is missing.
    """
    groundtruth_data = read_txt_file(groundtruth_path)
    llm_data = read_txt_file(llm_path)
    if groundtruth_data is None or llm_data is None:
        return None

    return truncate_to_common_length(llm_data.splitlines(), groundtruth_data.splitlines())


def load_pair(
    groundtruth_path: str, llm_path: str
) -> Optional[Tuple[List[str], List[str]]]:
    """Load a ground truth file and its prediction, dispatching on the suffix.

    :param groundtruth_path: Path to the ground truth file.
    :param llm_path: Path to the LLM answer.
    :return: The aligned predictions and references, or ``None`` when the
        prediction is missing.
    :raises ValueError: When the suffix is not one of
        :data:`SUPPORTED_SUFFIXES`.
    """
    if groundtruth_path.endswith(".json"):
        return load_json_pair(groundtruth_path, llm_path)
    if groundtruth_path.endswith(".txt"):
        return load_txt_pair(groundtruth_path, llm_path)
    raise ValueError(f"Unsupported ground truth file: {groundtruth_path}")


def list_groundtruth_files(groundtruth_dir: str) -> List[str]:
    """List the ground truth files of a directory, sorted by name.

    :param groundtruth_dir: The directory holding the ground truth.
    :return: The file names with a supported suffix.
    :rtype: list[str]
    """
    return sorted(
        name
        for name in os.listdir(groundtruth_dir)
        if name.endswith(SUPPORTED_SUFFIXES)
    )


def evaluate_file(
    groundtruth_path: str, llm_path: str, suite: MetricSuite
) -> Optional[Dict[str, float]]:
    """Score one LLM answer against its ground truth.

    :param groundtruth_path: Path to the ground truth file.
    :param llm_path: Path to the LLM answer.
    :param suite: The metrics to run.
    :return: One value per column of the suite, zeroed when the LLM returned
        nothing, or ``None`` when the answer file is missing.
    """
    pair = load_pair(groundtruth_path, llm_path)
    if pair is None:
        return None

    predictions, references = pair
    if not predictions:
        return suite.zero_scores()

    return suite.compute(predictions, references)


def write_results_csv(
    output_path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]
) -> None:
    """Write the evaluation results to a CSV file.

    :param output_path: Path of the CSV file to create.
    :param rows: The result rows.
    :param fieldnames: The column names, in order.
    """
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_directory(
    groundtruth_dir: str,
    llm_dir: str,
    suite: MetricSuite,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    """Score every ground truth file of a directory against its prediction.

    :param groundtruth_dir: The directory holding the ground truth.
    :param llm_dir: The directory holding the LLM answers, named identically.
    :param suite: The metrics to run.
    :param output_path: Where to write the CSV. Nothing is written when
        omitted.
    :param verbose: Print each file and its scores while running.
    :return: One row per scored file, with a leading ``file`` column.
    :rtype: list[dict]
    """
    rows: List[Dict[str, object]] = []

    for name in list_groundtruth_files(groundtruth_dir):
        groundtruth_path = os.path.join(groundtruth_dir, name)
        llm_path = os.path.join(llm_dir, name)
        if verbose:
            print(groundtruth_path, llm_path)

        scores = evaluate_file(groundtruth_path, llm_path, suite)
        if scores is None:
            continue

        row: Dict[str, object] = {"file": name}
        row.update(scores)
        rows.append(row)
        if verbose:
            print(row)

    if output_path is not None:
        write_results_csv(output_path, rows, ["file"] + suite.fields)
        if verbose:
            print(f"Results written to {output_path}")

    return rows
