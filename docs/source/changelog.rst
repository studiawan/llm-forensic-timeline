Changelog
=========

0.0.2
-----

First packaged release.

* Split the two scripts into the ``llm_forensic_timeline`` package: ``metrics``,
  ``evaluation``, ``runner``, and ``cli``.
* Added BERTScore through ``modern-bert-score``, with
  ``LazerLambda/ModernBERT-large-ModBERTScore-19`` and baseline rescaling, as
  the semantic-aware complement to BLEU and ROUGE.
* Added the ``llm-forensic-timeline`` command with the ``run`` and ``evaluate``
  subcommands.
* Added a pytest suite, Sphinx documentation, and GitHub Actions workflows for
  testing and for publishing to PyPI through OIDC Trusted Publishing.
