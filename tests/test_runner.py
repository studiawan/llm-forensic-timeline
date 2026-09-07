"""Tests for :mod:`llm_forensic_timeline.runner`."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_forensic_timeline.runner import (
    DEFAULT_MODEL,
    build_messages,
    create_assistant,
    list_prompt_files,
    load_prompt,
    run_task,
    run_timeline,
    save_generated_files,
    send_message,
    upload_timeline,
    wait_for_run,
)

PROMPT = {
    "task": "event-summarization",
    "type": "no-knowledge",
    "role": "I am a forensic investigator.",
    "instructions": "Summarize the high level events.",
    "output_format": "Export the results to a JSON file.",
    "goals": {"14-shutdown.csv": "Summarize the last shutdown event."},
}


def make_annotation(file_id="file-out", text="sandbox:/mnt/data/summary.json"):
    """Build a file_path annotation as the Assistants API returns it."""
    return SimpleNamespace(
        type="file_path", text=text, file_path=SimpleNamespace(file_id=file_id)
    )


def make_message(annotations=(), value="Here are the results."):
    """Build an assistant message carrying the given annotations."""
    text = SimpleNamespace(value=value, annotations=list(annotations))
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def make_client(run_statuses=("completed",), annotations=()):
    """Build a mock OpenAI client that walks through ``run_statuses``."""
    client = MagicMock()
    client.files.create.return_value = SimpleNamespace(id="file-in")
    client.files.content.return_value = SimpleNamespace(
        read=lambda: b'{"id": 1002}'
    )
    client.beta.assistants.create.return_value = SimpleNamespace(id="asst-1")
    client.beta.threads.create.return_value = SimpleNamespace(id="thread-1")
    client.beta.threads.runs.create.return_value = SimpleNamespace(
        id="run-1", status="queued"
    )
    client.beta.threads.runs.retrieve.side_effect = [
        SimpleNamespace(id="run-1", status=status) for status in run_statuses
    ]
    client.beta.threads.messages.list.return_value = SimpleNamespace(
        data=[make_message(annotations)]
    )
    return client


def test_load_prompt_reads_the_definition(tmp_path):
    path = tmp_path / "no-knowledge.json"
    path.write_text(json.dumps(PROMPT))

    assert load_prompt(str(path)) == PROMPT


def test_list_prompt_files_keeps_json_only_and_sorts(tmp_path):
    for name in ["b.json", "a.json", "notes.md"]:
        (tmp_path / name).write_text("{}")

    assert list_prompt_files(str(tmp_path)) == ["a.json", "b.json"]


def test_build_messages_sends_the_persona_then_the_task():
    messages = build_messages(PROMPT, "Summarize the last shutdown event.")

    assert messages[0] == PROMPT["role"]
    assert messages[1] == (
        "Summarize the last shutdown event.\n\n"
        "Summarize the high level events.\n\n"
        "Export the results to a JSON file."
    )


def test_upload_timeline_returns_the_file_id(tmp_path):
    csv_path = tmp_path / "14-shutdown.csv"
    csv_path.write_text("datetime,message\n")
    client = make_client()

    assert upload_timeline(client, str(csv_path)) == "file-in"
    assert client.files.create.call_args.kwargs["purpose"] == "assistants"


def test_create_assistant_attaches_the_file_to_the_code_interpreter():
    client = make_client()

    create_assistant(client, PROMPT["role"], "file-in")

    kwargs = client.beta.assistants.create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["instructions"] == PROMPT["role"]
    assert kwargs["tool_resources"] == {
        "code_interpreter": {"file_ids": ["file-in"]}
    }


def test_wait_for_run_polls_until_the_run_leaves_the_pending_states():
    client = make_client(run_statuses=("queued", "in_progress", "completed"))

    run = wait_for_run(client, "thread-1", "run-1", poll_interval=0)

    assert run.status == "completed"
    assert client.beta.threads.runs.retrieve.call_count == 3


def test_save_generated_files_writes_every_attachment(tmp_path):
    client = make_client()
    message = make_message([make_annotation()])

    saved = save_generated_files(client, message, str(tmp_path / "results"))

    assert saved == [str(tmp_path / "results" / "summary.json")]
    assert (tmp_path / "results" / "summary.json").read_bytes() == b'{"id": 1002}'


def test_save_generated_files_ignores_messages_without_attachments(tmp_path):
    client = make_client()

    assert save_generated_files(client, make_message(), str(tmp_path)) == []


def test_send_message_raises_when_the_run_does_not_complete(tmp_path):
    client = make_client(run_statuses=("failed",))

    with pytest.raises(RuntimeError, match="Run failed with status: failed"):
        send_message(
            client, "thread-1", "asst-1", "hello", str(tmp_path), poll_interval=0
        )


def test_run_timeline_sends_both_messages(tmp_path):
    csv_path = tmp_path / "14-shutdown.csv"
    csv_path.write_text("datetime,message\n")
    client = make_client(run_statuses=("completed", "completed"))

    run_timeline(
        client,
        PROMPT,
        str(csv_path),
        "Summarize the last shutdown event.",
        str(tmp_path / "results"),
        poll_interval=0,
    )

    sent = [
        call.kwargs["content"]
        for call in client.beta.threads.messages.create.call_args_list
    ]
    assert sent == build_messages(PROMPT, "Summarize the last shutdown event.")


def test_run_task_skips_a_timeline_that_is_not_on_disk(tmp_path, capsys):
    prompts_dir = tmp_path / "prompt" / "event-summarization"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "no-knowledge.json").write_text(json.dumps(PROMPT))
    (tmp_path / "dataset" / "event-summarization").mkdir(parents=True)
    client = make_client()

    saved = run_task(
        client,
        "event-summarization",
        dataset_dir=str(tmp_path / "dataset"),
        prompts_dir=str(tmp_path / "prompt"),
        results_dir=str(tmp_path / "results"),
    )

    assert saved == []
    assert "was not found" in capsys.readouterr().out
    client.beta.assistants.create.assert_not_called()


def test_run_task_returns_the_saved_answers(tmp_path):
    prompts_dir = tmp_path / "prompt" / "event-summarization"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "no-knowledge.json").write_text(json.dumps(PROMPT))
    dataset_dir = tmp_path / "dataset" / "event-summarization"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "14-shutdown.csv").write_text("datetime,message\n")
    client = make_client(
        run_statuses=("completed", "completed"), annotations=[make_annotation()]
    )

    saved = run_task(
        client,
        "event-summarization",
        dataset_dir=str(tmp_path / "dataset"),
        prompts_dir=str(tmp_path / "prompt"),
        results_dir=str(tmp_path / "results"),
        poll_interval=0,
    )

    assert saved == [str(tmp_path / "results" / "summary.json")] * 2
