"""Tests for :mod:`llm_forensic_timeline.evaluation`."""

import csv
import json

import pytest

from llm_forensic_timeline.evaluation import (
    as_entry_list,
    evaluate_directory,
    evaluate_file,
    list_groundtruth_files,
    load_json_pair,
    load_pair,
    load_txt_pair,
    read_json_file,
    read_txt_file,
    truncate_to_common_length,
    write_results_csv,
)


def test_read_json_file_returns_the_document(tmp_path):
    path = tmp_path / "event.json"
    path.write_text('{"type": "Shutdown time"}')

    assert read_json_file(str(path)) == {"type": "Shutdown time"}


def test_read_json_file_returns_none_when_missing(tmp_path):
    assert read_json_file(str(tmp_path / "absent.json")) is None


def test_read_txt_file_returns_none_when_missing(tmp_path):
    assert read_txt_file(str(tmp_path / "absent.txt")) is None


def test_as_entry_list_wraps_a_single_event():
    assert as_entry_list({"id": 1002}) == [{"id": 1002}]


def test_as_entry_list_keeps_a_list_of_events():
    assert as_entry_list([{"id": 1002}, {"id": 1003}]) == [{"id": 1002}, {"id": 1003}]


@pytest.mark.parametrize(
    "predictions, references, expected",
    [
        ([1, 2, 3], [1, 2], ([1, 2], [1, 2])),
        ([1], [1, 2, 3], ([1], [1])),
        ([1, 2], [3, 4], ([1, 2], [3, 4])),
        ([], [1, 2], ([], [])),
    ],
)
def test_truncate_to_common_length(predictions, references, expected):
    assert truncate_to_common_length(predictions, references) == expected


def test_load_json_pair_serialises_entries_back_to_json(tmp_path):
    event = {"id": 1002, "type": "Shutdown time"}
    (tmp_path / "gt.json").write_text(json.dumps([event]))
    (tmp_path / "llm.json").write_text(json.dumps(event))

    predictions, references = load_json_pair(
        str(tmp_path / "gt.json"), str(tmp_path / "llm.json")
    )

    assert predictions == [json.dumps(event)]
    assert references == [json.dumps(event)]


def test_load_json_pair_truncates_an_over_long_answer(tmp_path):
    (tmp_path / "gt.json").write_text(json.dumps([{"id": 1}]))
    (tmp_path / "llm.json").write_text(json.dumps([{"id": 1}, {"id": 2}]))

    predictions, references = load_json_pair(
        str(tmp_path / "gt.json"), str(tmp_path / "llm.json")
    )

    assert len(predictions) == len(references) == 1


def test_load_json_pair_returns_none_when_the_answer_is_missing(tmp_path):
    (tmp_path / "gt.json").write_text("[]")

    assert load_json_pair(str(tmp_path / "gt.json"), str(tmp_path / "llm.json")) is None


def test_load_txt_pair_aligns_line_counts(tmp_path):
    (tmp_path / "gt.txt").write_text("first entry\nsecond entry\n")
    (tmp_path / "llm.txt").write_text("first entry\nsecond entry\nthird entry\n")

    predictions, references = load_txt_pair(
        str(tmp_path / "gt.txt"), str(tmp_path / "llm.txt")
    )

    assert predictions == ["first entry", "second entry"]
    assert references == ["first entry", "second entry"]


def test_load_pair_dispatches_on_the_suffix(tmp_path):
    (tmp_path / "gt.txt").write_text("only line\n")
    (tmp_path / "llm.txt").write_text("only line\n")

    assert load_pair(str(tmp_path / "gt.txt"), str(tmp_path / "llm.txt")) == (
        ["only line"],
        ["only line"],
    )


def test_load_pair_rejects_an_unsupported_suffix(tmp_path):
    with pytest.raises(ValueError, match="Unsupported ground truth file"):
        load_pair(str(tmp_path / "gt.csv"), str(tmp_path / "llm.csv"))


def test_list_groundtruth_files_keeps_only_supported_suffixes(tmp_path):
    for name in ["b.json", "a.txt", "notes.md", "results.csv"]:
        (tmp_path / name).write_text("")

    assert list_groundtruth_files(str(tmp_path)) == ["a.txt", "b.json"]


def test_evaluate_file_scores_the_pair(tmp_path, fake_suite, fake_metric):
    (tmp_path / "gt.json").write_text(json.dumps([{"id": 1}, {"id": 2}]))
    (tmp_path / "llm.json").write_text(json.dumps([{"id": 1}, {"id": 2}]))

    scores = evaluate_file(
        str(tmp_path / "gt.json"), str(tmp_path / "llm.json"), fake_suite
    )

    assert scores == {"alpha": 2.0, "beta": 0.5}
    assert len(fake_metric.calls) == 1


def test_evaluate_file_returns_zeros_for_an_empty_answer(tmp_path, fake_suite, fake_metric):
    (tmp_path / "gt.json").write_text(json.dumps([{"id": 1}]))
    (tmp_path / "llm.json").write_text("[]")

    scores = evaluate_file(
        str(tmp_path / "gt.json"), str(tmp_path / "llm.json"), fake_suite
    )

    assert scores == {"alpha": 0, "beta": 0}
    assert fake_metric.calls == []


def test_evaluate_file_returns_none_for_a_missing_answer(tmp_path, fake_suite):
    (tmp_path / "gt.json").write_text("[]")

    assert (
        evaluate_file(str(tmp_path / "gt.json"), str(tmp_path / "llm.json"), fake_suite)
        is None
    )


def test_write_results_csv_creates_missing_directories(tmp_path):
    output_path = tmp_path / "nested" / "results.csv"

    write_results_csv(str(output_path), [{"file": "a.json", "alpha": 1}], ["file", "alpha"])

    assert output_path.read_text().splitlines() == ["file,alpha", "a.json,1"]


def test_evaluate_directory_returns_one_row_per_file(timeline_dirs, fake_suite):
    groundtruth_dir, llm_dir = timeline_dirs

    rows = evaluate_directory(
        str(groundtruth_dir), str(llm_dir), fake_suite, verbose=False
    )

    assert rows == [{"file": "14-shutdown.json", "alpha": 1.0, "beta": 0.5}]


def test_evaluate_directory_skips_files_without_an_answer(timeline_dirs, fake_suite):
    groundtruth_dir, _ = timeline_dirs
    (groundtruth_dir / "15-search.json").write_text("[]")

    rows = evaluate_directory(
        str(groundtruth_dir), str(groundtruth_dir.parent / "empty"), fake_suite, verbose=False
    )

    assert rows == []


def test_evaluate_directory_writes_the_csv_header_in_suite_order(
    tmp_path, timeline_dirs, fake_suite
):
    groundtruth_dir, llm_dir = timeline_dirs
    output_path = tmp_path / "results.csv"

    evaluate_directory(
        str(groundtruth_dir),
        str(llm_dir),
        fake_suite,
        output_path=str(output_path),
        verbose=False,
    )

    with open(output_path, newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["file", "alpha", "beta"]
    assert rows[1][0] == "14-shutdown.json"
