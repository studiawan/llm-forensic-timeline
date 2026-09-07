Installation
============

From PyPI
---------

.. code-block:: console

   pip install llm-forensic-timeline[all]

The base install pulls only BLEU and ROUGE. The extras add the optional halves:

=============  ========================================================
Extra          Adds
=============  ========================================================
``bertscore``  ``modern-bert-score``, needed for the semantic metric
``llm``        ``openai`` and ``python-dotenv``, needed to run a task
``all``        both of the above
``test``       ``pytest``
``docs``       Sphinx and its theme
=============  ========================================================

.. note::

   ``modern-bert-score`` requires Python 3.12 or newer. On an older
   interpreter the ``bertscore`` extra is skipped, and the evaluation still
   runs with ``--no-bertscore``.

From source
-----------

.. code-block:: console

   git clone https://github.com/studiawan/llm-forensic-timeline
   cd llm-forensic-timeline
   python3.13 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[all,test]"

On Windows, activate with ``.venv\Scripts\activate``.

Dataset
-------

1. Open the dataset repository on `Zenodo <https://zenodo.org/records/15493424>`_.
2. Download ``scenario-1.zip`` and unzip it.
3. Copy every file from ``scenario-1/timeline/events`` into
   ``dataset/event-summarization``.

API key
-------

Running a task calls the OpenAI API. Put the key in a ``.env`` file:

.. code-block:: console

   OPENAI_API_KEY=sk-proj-xxx

The file is looked up next to the calling script and upwards, so a ``.env`` in
the repository root or in ``src/`` both work. Pass ``--env-file`` to point at a
different one.
