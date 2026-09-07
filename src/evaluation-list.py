"""Score the LLM answers of one task against the ground truth.

Thin wrapper kept for the workflow documented in the paper. Edit the two
variables below and run it from the directory that holds the task directory.
Everything it does is also available as ``llm-forensic-timeline evaluate``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_forensic_timeline.evaluation import evaluate_directory  # noqa: E402
from llm_forensic_timeline.metrics import build_default_suite  # noqa: E402

# define the type and prediction
task = 'event-reconstruction'  # 'suspicious-keyword' # 'grep'
prediction = 'chatgpt-no-knowledge-multiple'  # 'chatgpt-with-knowledge-multiple' # 'chatgpt-no-knowledge' # 'chatgpt-with-knowledge'

# define the directories
groundtruth_dir = f'./{task}/ground-truth-multiple/'  # ground-truth-multiple
llm_dir = f'./{task}/{prediction}'

# create a csv file to store the results
csv_file = f'./{task}/{prediction}-results.csv'


def main():
    """Evaluate the configured task and write the results CSV."""
    print(csv_file)
    suite = build_default_suite()
    evaluate_directory(groundtruth_dir, llm_dir, suite, output_path=csv_file)


if __name__ == "__main__":
    main()
