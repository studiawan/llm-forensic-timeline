Quick Start
===========

Send a timeline to the LLM
--------------------------

.. code-block:: console

   llm-forensic-timeline run --task event-summarization

The command reads every prompt in ``prompt/event-summarization/``, uploads the
CSV timelines those prompts name from ``dataset/event-summarization/``, and
saves each file the assistant generates into ``results/``.

Score the answers
-----------------

.. code-block:: console

   llm-forensic-timeline evaluate \
       --groundtruth-dir event-reconstruction/ground-truth-multiple \
       --llm-dir event-reconstruction/chatgpt-with-knowledge-multiple \
       --output event-reconstruction/chatgpt-with-knowledge-multiple-results.csv

Every ground truth file is paired with the file of the same name in the answer
directory. The resulting CSV holds one row per file:

.. code-block:: text

   file,bleu,brevity_penalty,length_ratio,translation_length,reference_length,rouge1,rouge2,rougeL,rougeLsum,bertscore_precision,bertscore_recall,bertscore_f1
   14-shutdown.json,0.8304,1.0,1.0286,72,70,0.7949,0.6477,0.7949,0.7949,0.8758,0.8769,0.8760

From Python
-----------

.. code-block:: python

   from llm_forensic_timeline import build_default_suite, evaluate_directory

   suite = build_default_suite()
   rows = evaluate_directory(
       "event-reconstruction/ground-truth-multiple",
       "event-reconstruction/chatgpt-with-knowledge-multiple",
       suite,
       output_path="results.csv",
   )

Drop BERTScore when you only need the lexical metrics, which avoids loading a
transformer model:

.. code-block:: python

   suite = build_default_suite(with_bertscore=False)
