# llm-forensic-timeline

[![Tests](https://github.com/studiawan/llm-forensic-timeline/actions/workflows/test.yml/badge.svg)](https://github.com/studiawan/llm-forensic-timeline/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/llm-forensic-timeline/badge/?version=latest)](https://llm-forensic-timeline.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/llm-forensic-timeline.svg)](https://pypi.org/project/llm-forensic-timeline/)

A work in progress repository for DFRWS APAC 2025 paper: [A standardized methodology and dataset for evaluating LLM-based digital forensic timeline analysis](https://www.sciencedirect.com/science/article/pii/S2666281725001222).

The package covers the two halves of the methodology:

- **Run** — upload a log2timeline/Plaso CSV timeline to an LLM, replay the prompts of a task, and save every answer the model generates.
- **Evaluate** — score those answers against a manually validated ground truth with BLEU, ROUGE, and BERTScore, and write one CSV row per file.

Documentation: <https://llm-forensic-timeline.readthedocs.io>

# Requirements

1. Create a virtual environment using `venv` (Python 3.10 or newer; BERTScore additionally needs 3.12+):

   `python3.13 -m venv .venv`

2. Activate the environment:

    `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)

3. Install the package:

   `pip install -e ".[all]"`

   Or, without cloning: `pip install llm-forensic-timeline[all]`

4. Open dataset repository on [Zenodo](https://zenodo.org/records/15493424)

5. Download `scenario-1.zip` and unzip it

6. Copy all files from `scenario-1/timeline/events` directory to `dataset/event-summarization` directory

The extras split the optional halves: `bertscore` adds `modern-bert-score`, `llm` adds `openai` and `python-dotenv`, and `all` adds both. `requirements.txt` still holds the full environment used for the paper.

# How to run

1. Add `.env` file in `src` directory. Add your OpenAI API keys in the `.env` file, something like: `OPENAI_API_KEY=sk-proj-xxx`

2. Run the task

   `llm-forensic-timeline run --task event-summarization`

   The original script still works: `python src/run.py`

# How to evaluate

`llm-forensic-timeline evaluate` compares the LLM answers against the ground truth and writes the scores to a CSV file:

```
llm-forensic-timeline evaluate \
    --groundtruth-dir event-reconstruction/ground-truth-multiple \
    --llm-dir event-reconstruction/chatgpt-with-knowledge-multiple \
    --output event-reconstruction/chatgpt-with-knowledge-multiple-results.csv
```

The original script still works: edit the `task` and `prediction` variables at the top of `src/evaluation-list.py`, then run it from the directory that contains the task directory.

Three metrics are reported per file:

1. **BLEU** and **ROUGE** (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum) through the Hugging Face `evaluate` library. Both are n-gram overlap metrics, so they only measure lexical similarity.

2. **BERTScore** through [`modern-bert-score`](https://pypi.org/project/modern-bert-score/), which is semantic-aware and therefore credits an answer that is worded differently but carries the same meaning. This addresses the limitation of BLEU and ROUGE discussed in Sec. 4.4.2 of the paper. The columns `bertscore_precision`, `bertscore_recall`, and `bertscore_f1` hold the mean over all entries of a file.

Notes on BERTScore:

- The model is `LazerLambda/ModernBERT-large-ModBERTScore-19`, the strongest checkpoint shipped with the package. It is the largest one, and its 8192-token context keeps long JSON entries of a forensic timeline intact, while the RoBERTa checkpoints truncate at 512 tokens. The weights (about 1.6 GB) are downloaded from Hugging Face on the first run and cached afterwards.
- Scores are rescaled with the baseline of the model, as recommended by the BERTScore authors. Without rescaling, scores of unrelated texts sit near 0.8; with rescaling, they sit near 0 and can be slightly negative, which makes the results easier to read.
- The script selects CUDA or Apple Silicon (MPS) automatically and falls back to the CPU. Evaluating a large timeline on the CPU is slow.
- Pass `--no-bertscore` to score with BLEU and ROUGE only, without loading a transformer model.

# Development

Run the test suite:

```
pip install -e ".[test]"
pytest
```

Build the documentation locally:

```
pip install -r docs/requirements.txt
python -m sphinx -W -b html docs/source docs/_build/html
```

The `Run Tests` workflow runs pytest on Python 3.10 through 3.13 and builds the documentation on every push and pull request to `main`. Pushing a `v*.*.*` tag builds the distributions and publishes them to PyPI through OIDC Trusted Publishing, without any stored credentials.
