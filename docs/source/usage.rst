Detailed Usage
==============

Command line reference
----------------------

``llm-forensic-timeline run``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Drives an LLM over the timelines of one task.

==========================  ==========================================================
Flag                        Description
==========================  ==========================================================
``-t`` / ``--task``         Task name, e.g. ``event-summarization``. Required.
``-d`` / ``--dataset-dir``  Root holding ``<dataset-dir>/<task>/``. Default ``dataset``.
``-p`` / ``--prompts-dir``  Root holding ``<prompts-dir>/<task>/``. Default ``prompt``.
``-o`` / ``--results-dir``  Where to save the answers. Default ``results``.
``-m`` / ``--model``        The OpenAI model to run. Default ``gpt-4o``.
``--env-file``              Path to the ``.env`` holding ``OPENAI_API_KEY``.
==========================  ==========================================================

``llm-forensic-timeline evaluate``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scores saved answers against the ground truth.

===============================  =====================================================
Flag                             Description
===============================  =====================================================
``-g`` / ``--groundtruth-dir``   Directory of ground truth files. Required.
``-l`` / ``--llm-dir``           Directory of LLM answers. Required.
``-o`` / ``--output``            Where to write the results CSV.
``--no-bertscore``               Score with BLEU and ROUGE only.
``--bertscore-model``            The BERTScore checkpoint to embed with.
``--device``                     ``cuda``, ``mps``, or ``cpu``. Detected when omitted.
===============================  =====================================================

Prompt files
------------

A prompt is a JSON document with five keys. ``goals`` maps a CSV timeline to
the goal pursued on it, which is how one prompt covers several timelines:

.. code-block:: json

   {
     "task": "event-summarization",
     "type": "no-knowledge-single-event",
     "role": "I am a forensic investigator.",
     "instructions": "Summarize the high level events of the timeline.",
     "output_format": "Export the results to a JSON file.",
     "goals": {
       "14-shutdown.csv": "Summarize the last shutdown event."
     }
   }

Two messages are sent per timeline: the persona (``role``), then ``goal``,
``instructions``, and ``output_format`` joined by blank lines.

Ground truth formats
--------------------

The evaluation dispatches on the file suffix.

``.json``
    Each entry is serialised back to JSON and compared as one string, so the
    metrics see the structure and the values, not just the free text. A single
    high-level event stored as an object is wrapped into a one-entry list.

``.txt``
    The file is compared line by line, which is what the grep task produces.

In both cases the longer side is truncated to the length of the shorter one,
because an LLM regularly returns more or fewer entries than the ground truth
holds.

Files with an unsupported suffix are skipped. A ground truth file whose answer
is missing is skipped as well; an answer that decodes to an empty list yields a
zeroed row so the file still appears in the CSV.
