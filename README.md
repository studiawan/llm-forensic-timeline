# llm-forensic-timeline
A work in progress repository for DFRWS APAC 2025 paper: [A standardized methodology and dataset for evaluating LLM-based digital forensic timeline analysis](https://www.sciencedirect.com/science/article/pii/S2666281725001222).

# Requirements

1. Create a virtual environment using Anaconda:

   `conda create --name llm-timeline python=3.13`

2. Activate the environment:

    `conda activate llm-timeline`

3. Install `openai` package: 

   `pip install openai`

4. Download `scenario-1.zip` and unzip it

5. Copy all files from `scenario-1/timeline/events` directory to `dataset/event-summarization` directory

# How to run

1. Add `.env` file in `src` directory. Add your OpenAI API keys in the `.env` file, something like: `OPENAI_API_KEY=sk-proj-xxx`

2. Run the script

   `python src/run.py`
