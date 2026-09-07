"""Driving an LLM over a forensic timeline.

The runner uploads a Plaso CSV timeline to the OpenAI Assistants API, replays
the prompts of a task, and saves every file the assistant generates. Those
files are the answers that :mod:`llm_forensic_timeline.evaluation` scores.

The OpenAI SDK is imported lazily so that the evaluation half of the package
stays usable without it.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

#: The model used for the case study in the paper.
DEFAULT_MODEL = "gpt-4o"

#: Statuses that mean the assistant is still working.
PENDING_STATUSES = ("queued", "in_progress")


def build_client(env_path: Optional[str] = None) -> Any:
    """Create an OpenAI client, reading the API key from a ``.env`` file.

    :param env_path: Path to the ``.env`` file. The nearest one is used when
        omitted.
    :return: A configured ``openai.OpenAI`` client.
    """
    import openai
    from dotenv import load_dotenv

    load_dotenv(env_path)
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_prompt(prompt_path: str) -> Dict[str, Any]:
    """Read one prompt definition.

    A prompt file holds the persona (``role``), the per-timeline ``goals``, the
    ``instructions``, and the required ``output_format``.

    :param prompt_path: Path to the JSON prompt file.
    :return: The decoded prompt.
    :rtype: dict
    """
    with open(prompt_path, "r") as handle:
        return json.load(handle)


def list_prompt_files(prompts_dir: str) -> List[str]:
    """List the JSON prompt files of a directory, sorted by name.

    :param prompts_dir: The directory holding the prompts.
    :return: The file names.
    :rtype: list[str]
    """
    return sorted(
        name for name in os.listdir(prompts_dir) if name.endswith(".json")
    )


def build_messages(prompt: Dict[str, Any], goal: str) -> List[str]:
    """Build the messages sent to the assistant for one timeline.

    :param prompt: A prompt definition, as returned by :func:`load_prompt`.
    :param goal: The goal attached to the timeline being analysed.
    :return: The persona message followed by the task message.
    :rtype: list[str]
    """
    task_message = "\n\n".join(
        [goal, prompt["instructions"], prompt["output_format"]]
    )
    return [prompt["role"], task_message]


def upload_timeline(client: Any, csv_path: str) -> str:
    """Upload a Plaso CSV timeline so the assistant can read it.

    :param client: The OpenAI client.
    :param csv_path: Path to the CSV timeline.
    :return: The identifier of the uploaded file.
    :rtype: str
    """
    print(f"Uploading forensic timeline in CSV: {csv_path}...")
    with open(csv_path, "rb") as handle:
        file_object = client.files.create(file=handle, purpose="assistants")
    print(f"File uploaded successfully. File ID: {file_object.id}")
    return file_object.id


def create_assistant(
    client: Any, instructions: str, file_id: str, model: str = DEFAULT_MODEL
) -> Any:
    """Create an assistant with the code interpreter over the uploaded file.

    :param client: The OpenAI client.
    :param instructions: The persona the assistant adopts.
    :param file_id: The identifier returned by :func:`upload_timeline`.
    :param model: The model to run.
    :return: The created assistant.
    """
    return client.beta.assistants.create(
        name="Forensic timeline analyst",
        instructions=instructions,
        model=model,
        tools=[{"type": "code_interpreter"}],
        tool_resources={"code_interpreter": {"file_ids": [file_id]}},
    )


def wait_for_run(
    client: Any, thread_id: str, run_id: str, poll_interval: float = 1.0
) -> Any:
    """Poll a run until the assistant stops working.

    :param client: The OpenAI client.
    :param thread_id: The thread the run belongs to.
    :param run_id: The run to wait for.
    :param poll_interval: Seconds to sleep between polls.
    :return: The finished run, whatever its status.
    """
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    while run.status in PENDING_STATUSES:
        time.sleep(poll_interval)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    return run


def save_generated_files(client: Any, message: Any, results_dir: str) -> List[str]:
    """Download every file the assistant attached to a message.

    The answers are evaluated from these downloaded files, not from the text of
    the reply.

    :param client: The OpenAI client.
    :param message: The assistant message to inspect.
    :param results_dir: The directory to write the files into.
    :return: The paths of the saved files.
    :rtype: list[str]
    """
    text_content = message.content[0].text
    annotations = getattr(text_content, "annotations", None) or []

    saved_paths: List[str] = []
    for annotation in annotations:
        if annotation.type != "file_path":
            continue

        file_id = annotation.file_path.file_id
        file_name = annotation.text.split("/")[-1]
        print(f"\nAssistant generated a file: '{file_name}' ({file_id})")

        print("Downloading file...")
        content = client.files.content(file_id).read()

        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, file_name)
        with open(output_path, "wb") as handle:
            handle.write(content)
        print(f"File '{file_name}' downloaded successfully.")
        saved_paths.append(output_path)

    return saved_paths


def send_message(
    client: Any,
    thread_id: str,
    assistant_id: str,
    message: str,
    results_dir: str,
    poll_interval: float = 1.0,
) -> List[str]:
    """Send one message, wait for the answer, and save its attachments.

    :param client: The OpenAI client.
    :param thread_id: The conversation thread.
    :param assistant_id: The assistant to run.
    :param message: The message to send.
    :param results_dir: The directory to write generated files into.
    :param poll_interval: Seconds to sleep between polls.
    :return: The paths of the saved files.
    :rtype: list[str]
    :raises RuntimeError: When the run does not complete.
    """
    print(f"User: {message}")
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=message
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=assistant_id
    )
    print("Assistant is thinking...")
    run = wait_for_run(client, thread_id, run.id, poll_interval=poll_interval)

    if run.status != "completed":
        raise RuntimeError(f"Run failed with status: {run.status}")

    print("Assistant completed successfully.")
    messages = client.beta.threads.messages.list(thread_id=thread_id, limit=1)
    assistant_message = messages.data[0]
    print(f"Assistant: {assistant_message.content[0].text.value}")

    return save_generated_files(client, assistant_message, results_dir)


def run_timeline(
    client: Any,
    prompt: Dict[str, Any],
    csv_path: str,
    goal: str,
    results_dir: str,
    model: str = DEFAULT_MODEL,
    poll_interval: float = 1.0,
) -> List[str]:
    """Run one prompt against one timeline.

    :param client: The OpenAI client.
    :param prompt: A prompt definition, as returned by :func:`load_prompt`.
    :param csv_path: Path to the CSV timeline.
    :param goal: The goal attached to that timeline.
    :param results_dir: The directory to write generated files into.
    :param model: The model to run.
    :param poll_interval: Seconds to sleep between polls.
    :return: The paths of every file saved for this timeline.
    :rtype: list[str]
    """
    file_id = upload_timeline(client, csv_path)

    print("Creating Assistant and a new conversation Thread...")
    assistant = create_assistant(client, prompt["role"], file_id, model=model)
    thread = client.beta.threads.create()
    print(f"Assistant ({assistant.id}) and Thread ({thread.id}) created.")

    saved_paths: List[str] = []
    for message in build_messages(prompt, goal):
        saved_paths.extend(
            send_message(
                client,
                thread.id,
                assistant.id,
                message,
                results_dir,
                poll_interval=poll_interval,
            )
        )
    return saved_paths


def run_task(
    client: Any,
    task: str,
    dataset_dir: str = "dataset",
    prompts_dir: str = "prompt",
    results_dir: str = "results",
    model: str = DEFAULT_MODEL,
    poll_interval: float = 1.0,
) -> List[str]:
    """Run every prompt of a task against every timeline it names.

    :param client: The OpenAI client.
    :param task: The task name, e.g. ``"event-summarization"``.
    :param dataset_dir: The root holding ``<dataset_dir>/<task>/``.
    :param prompts_dir: The root holding ``<prompts_dir>/<task>/``.
    :param results_dir: The directory to write generated files into.
    :param model: The model to run.
    :param poll_interval: Seconds to sleep between polls.
    :return: The paths of every file saved for the task.
    :rtype: list[str]
    """
    task_dataset_dir = os.path.join(dataset_dir, task)
    task_prompts_dir = os.path.join(prompts_dir, task)

    saved_paths: List[str] = []
    for prompt_file in list_prompt_files(task_prompts_dir):
        prompt_path = os.path.join(task_prompts_dir, prompt_file)
        print("JSON Path:", prompt_path)
        prompt = load_prompt(prompt_path)

        for file_name, goal in prompt["goals"].items():
            csv_path = os.path.join(task_dataset_dir, file_name)
            print(f"File Name: {csv_path}, Goal: {goal}")

            if not os.path.isfile(csv_path):
                print(f"Error: The file '{csv_path}' was not found.")
                continue

            saved_paths.extend(
                run_timeline(
                    client,
                    prompt,
                    csv_path,
                    goal,
                    results_dir,
                    model=model,
                    poll_interval=poll_interval,
                )
            )
    return saved_paths
