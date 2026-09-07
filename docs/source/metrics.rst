Evaluation Metrics
==================

BLEU and ROUGE
--------------

Both come from the Hugging Face ``evaluate`` library and both measure n-gram
overlap. They are precise and deterministic, but they score wording, not
meaning: an answer that says the same thing in different words scores low.

Sec. 4.3.1 of the paper reports exactly that. When ChatGPT reproduced a grep
result correctly but dropped the commas of the CSV, BLEU fell while the content
was intact.

BERTScore
---------

BERTScore embeds both sides with a transformer and matches tokens by cosine
similarity, so it credits a semantically equivalent answer. It is the
semantic-aware metric recommended in Sec. 4.4.2 of the paper, and it is
computed here through `modern-bert-score
<https://pypi.org/project/modern-bert-score/>`_.

Model choice
^^^^^^^^^^^^

The default checkpoint is ``LazerLambda/ModernBERT-large-ModBERTScore-19``, the
strongest one shipped with the package. It is the largest, and its 8192-token
context keeps the long JSON entries of a forensic timeline intact, while the
RoBERTa checkpoints truncate at 512 tokens. The weights (about 1.6 GB) are
downloaded from Hugging Face on the first run and cached afterwards.

Baseline rescaling
^^^^^^^^^^^^^^^^^^

Scores are rescaled with the baseline of the model, as recommended by the
BERTScore authors. Without rescaling, unrelated texts still score near 0.8;
with rescaling they sit near 0 and can be slightly negative, which makes the
results easier to read.

Aggregation
^^^^^^^^^^^

The scorer returns one ``(precision, recall, f1)`` triple per pair. The triples
are averaged so that a file yields a single score, the same way BLEU and ROUGE
are aggregated. Pairs where either side is blank are dropped first, because an
empty string carries no token to match.

Device
^^^^^^

CUDA is used when available, then Apple Silicon (MPS), then the CPU. Scoring a
large timeline on the CPU is slow.

Reading the columns
-------------------

===========================  ==============================================
Column                       Meaning
===========================  ==============================================
``bleu``                     BLEU score, 0 to 1, higher is better
``brevity_penalty``          BLEU penalty for an answer shorter than the reference
``length_ratio``             Candidate length over reference length
``translation_length``       Token count of the answer
``reference_length``         Token count of the ground truth
``rouge1`` / ``rouge2``      Unigram and bigram overlap
``rougeL`` / ``rougeLsum``   Longest common subsequence overlap
``bertscore_precision``      Mean rescaled BERTScore precision
``bertscore_recall``         Mean rescaled BERTScore recall
``bertscore_f1``             Mean rescaled BERTScore F1
===========================  ==============================================
