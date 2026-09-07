"""Tests for :mod:`llm_forensic_timeline.cli`."""

import pytest

from llm_forensic_timeline import cli
from llm_forensic_timeline.metrics import DEFAULT_BERTSCORE_MODEL, BertScoreMetric
from llm_forensic_timeline.runner import DEFAULT_MODEL


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_run_defaults_follow_the_repository_layout():
    args = cli.build_parser().parse_args(["run", "--task", "event-summarization"])

    assert args.command == "run"
    assert args.dataset_dir == "dataset"
    assert args.prompts_dir == "prompt"
    assert args.results_dir == "results"
    assert args.model == DEFAULT_MODEL


def test_evaluate_requires_both_directories():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["evaluate", "--groundtruth-dir", "gt"])


def test_evaluate_defaults_to_scoring_with_bertscore():
    args = cli.build_parser().parse_args(
        ["evaluate", "--groundtruth-dir", "gt", "--llm-dir", "llm"]
    )

    assert args.no_bertscore is False
    assert args.bertscore_model == DEFAULT_BERTSCORE_MODEL
    assert args.device is None


def test_evaluate_command_reports_a_missing_ground_truth_directory(tmp_path, capsys):
    args = cli.build_parser().parse_args(
        [
            "evaluate",
            "--groundtruth-dir",
            str(tmp_path / "absent"),
            "--llm-dir",
            str(tmp_path),
        ]
    )

    assert cli.evaluate_command(args) == 1
    assert "Ground truth directory not found" in capsys.readouterr().out


def test_evaluate_command_reports_a_missing_answer_directory(tmp_path, capsys):
    args = cli.build_parser().parse_args(
        [
            "evaluate",
            "--groundtruth-dir",
            str(tmp_path),
            "--llm-dir",
            str(tmp_path / "absent"),
        ]
    )

    assert cli.evaluate_command(args) == 1
    assert "LLM answer directory not found" in capsys.readouterr().out


def test_main_evaluates_the_given_directories(monkeypatch, timeline_dirs, tmp_path):
    groundtruth_dir, llm_dir = timeline_dirs
    recorded = {}

    def fake_evaluate_directory(groundtruth, llm, suite, output_path=None):
        recorded.update(
            groundtruth=groundtruth, llm=llm, suite=suite, output_path=output_path
        )
        return []

    monkeypatch.setattr(cli, "evaluate_directory", fake_evaluate_directory)

    exit_code = cli.main(
        [
            "evaluate",
            "--groundtruth-dir",
            str(groundtruth_dir),
            "--llm-dir",
            str(llm_dir),
            "--output",
            str(tmp_path / "results.csv"),
            "--no-bertscore",
        ]
    )

    assert exit_code == 0
    assert recorded["groundtruth"] == str(groundtruth_dir)
    assert recorded["output_path"] == str(tmp_path / "results.csv")
    assert not any(
        isinstance(metric, BertScoreMetric) for metric in recorded["suite"].metrics
    )


def test_main_runs_the_task_with_the_parsed_options(monkeypatch, tmp_path):
    recorded = {}

    monkeypatch.setattr(cli, "evaluate_directory", lambda *a, **k: [])
    monkeypatch.setattr(
        "llm_forensic_timeline.runner.build_client", lambda env_path=None: "client"
    )
    monkeypatch.setattr(
        "llm_forensic_timeline.runner.run_task",
        lambda client, task, **kwargs: recorded.update(
            client=client, task=task, **kwargs
        ),
    )

    exit_code = cli.main(
        [
            "run",
            "--task",
            "event-summarization",
            "--results-dir",
            str(tmp_path / "results"),
            "--model",
            "gpt-4o-mini",
        ]
    )

    assert exit_code == 0
    assert recorded["client"] == "client"
    assert recorded["task"] == "event-summarization"
    assert recorded["model"] == "gpt-4o-mini"
    assert recorded["results_dir"] == str(tmp_path / "results")
