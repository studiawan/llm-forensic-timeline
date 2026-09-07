"""Command line interface of the package.

Two subcommands mirror the two halves of the methodology: ``run`` drives the
LLM over a timeline, and ``evaluate`` scores the answers it produced against
the ground truth.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from .evaluation import evaluate_directory
from .metrics import DEFAULT_BERTSCORE_MODEL, build_default_suite
from .runner import DEFAULT_MODEL


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    :return: The parser, with the ``run`` and ``evaluate`` subcommands.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="llm-forensic-timeline",
        description=(
            "Standardized evaluation of LLM-based digital forensic timeline "
            "analysis."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Send a forensic timeline and its prompts to an LLM"
    )
    run_parser.add_argument(
        "-t", "--task", required=True, help="Task name, e.g. event-summarization"
    )
    run_parser.add_argument(
        "-d", "--dataset-dir", default="dataset", help="Root of the CSV timelines"
    )
    run_parser.add_argument(
        "-p", "--prompts-dir", default="prompt", help="Root of the JSON prompts"
    )
    run_parser.add_argument(
        "-o", "--results-dir", default="results", help="Where to save the answers"
    )
    run_parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, help="The OpenAI model to run"
    )
    run_parser.add_argument(
        "--env-file", default=None, help="Path to the .env holding OPENAI_API_KEY"
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Score LLM answers against the ground truth"
    )
    evaluate_parser.add_argument(
        "-g", "--groundtruth-dir", required=True, help="Directory of ground truth files"
    )
    evaluate_parser.add_argument(
        "-l", "--llm-dir", required=True, help="Directory of LLM answers"
    )
    evaluate_parser.add_argument(
        "-o", "--output", default=None, help="Where to write the results CSV"
    )
    evaluate_parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Score with BLEU and ROUGE only, without loading a transformer model",
    )
    evaluate_parser.add_argument(
        "--bertscore-model",
        default=DEFAULT_BERTSCORE_MODEL,
        help="The BERTScore checkpoint to embed with",
    )
    evaluate_parser.add_argument(
        "--device",
        default=None,
        help="Device for BERTScore: cuda, mps, or cpu. Detected when omitted",
    )

    return parser


def run_command(args: argparse.Namespace) -> int:
    """Handle the ``run`` subcommand.

    :param args: The parsed arguments.
    :return: The process exit code.
    :rtype: int
    """
    from .runner import build_client, run_task

    client = build_client(args.env_file)
    run_task(
        client,
        args.task,
        dataset_dir=args.dataset_dir,
        prompts_dir=args.prompts_dir,
        results_dir=args.results_dir,
        model=args.model,
    )
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    """Handle the ``evaluate`` subcommand.

    :param args: The parsed arguments.
    :return: The process exit code.
    :rtype: int
    """
    if not os.path.isdir(args.groundtruth_dir):
        print(f"[!] Ground truth directory not found: {args.groundtruth_dir}")
        return 1
    if not os.path.isdir(args.llm_dir):
        print(f"[!] LLM answer directory not found: {args.llm_dir}")
        return 1

    suite = build_default_suite(
        with_bertscore=not args.no_bertscore,
        bertscore_model=args.bertscore_model,
        device=args.device,
    )
    evaluate_directory(
        args.groundtruth_dir, args.llm_dir, suite, output_path=args.output
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the ``llm-forensic-timeline`` command.

    :param argv: The argument list. ``sys.argv`` is used when omitted.
    :return: The process exit code.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_command(args)
    return evaluate_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
