"""Send the forensic timelines of a task to ChatGPT.

Thin wrapper kept for the workflow documented in the paper. Everything it does
is also available as ``llm-forensic-timeline run --task event-summarization``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_forensic_timeline.runner import build_client, run_task  # noqa: E402

# list of tasks
tasks = ["event-summarization"]


def main():
    """Run every task against the OpenAI Assistants API."""
    client = build_client()
    for task in tasks:
        run_task(client, task)


if __name__ == "__main__":
    main()
