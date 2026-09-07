llm-forensic-timeline
=====================

Standardized evaluation of LLM-based digital forensic timeline analysis.

This package implements the methodology of the DFRWS APAC 2025 paper `Towards a
standardized methodology and dataset for evaluating LLM-based digital forensic
timeline analysis <https://doi.org/10.1016/j.fsidi.2025.301982>`_. It covers the
two halves of that methodology:

* **Run.** Upload a log2timeline/Plaso CSV timeline to an LLM, replay the
  prompts of a task, and save every answer the model generates.
* **Evaluate.** Score those answers against a manually validated ground truth
  with BLEU, ROUGE, and BERTScore, and write one CSV row per file.

The forensic timeline dataset and ground truth are published separately on
`Zenodo <https://zenodo.org/records/15493424>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   usage
   metrics
   api
   changelog
   license

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
