# `orality/` — Orality Level estimator g(·) + evaluation suite

This package implements the seven-feature **Orality Level** estimator described
in §3 of the paper, and the L-Acc / BERTScore / BLEU / ROUGE / PPL / O-score
metrics used throughout the paper.

## Public API

```python
from orality import stat_all_metrics, eval_oscore_and_level

stats              = stat_all_metrics("OK so, um, what's going on?")
o_score, level     = eval_oscore_and_level(stats)
#  stats: {"filler": ..., "interact": ..., "stance": ..., "simple": ...,
#          "repeat": ..., "PID": ..., "NVR": ..., "effective_word_count": ...}
#  level: int in {0, 1, 2, 3, 4, 5}
```

The estimator agrees with human pairwise judgments on **91 %** of held-out
pairs and **outperforms** an ensemble of frontier LLM judges
(DeepSeek-R1 / Gemini-3-Flash / GPT-4o) and four Bradley-Terry reward models
trained on 10 K LLM-judged pairs (Appendix B of the paper).

## Module map

| File | What it does | Paper section |
| --- | --- | --- |
| `stat_metric8.py` | Extracts the seven orality features (+ effective word count) | §3.2 |
| `vocabulary.py` | The English oral lexicon (fillers, interaction markers, stance/hedges, lexical reduction, repetition phrases) | §3.2 |
| `ideadensity/` | Propositional Idea Density (PID) sub-package; depends only on spaCy | §3.2, Appendix A.1 |
| `eval_oscore_level.py` | EWM weighting + PCA + KMeans cluster→level mapping; loads `oral_level_model.pkl` | §3.3 |
| `oral_level_model.pkl` | Pre-fit EWM weights, PCA basis, KMeans cluster→level map (used in all paper numbers) | §3.3 |
| `eval_predictions.py` | End-to-end evaluator: BLEU / ROUGE / BERTScore F1 / PPL / O-score MAE / Level Accuracy / Macro-F1 | §5 |

## Why feature-based and not a learned classifier?

§3 of the paper argues that orality is an **ordinal, relative** psychometric
attribute (Stevens 1946, Thurstone 1927). Asking annotators which of two
texts is more spoken is reliable; asking them to assign an absolute level is
not. A learned classifier inherits this rater noise, while our feature-based
estimator anchors level boundaries in five decades of register linguistics
(Akinnaso 1982; Biber 1991; van Compernolle 2018).

Empirically:

| Estimator | Pair-Acc vs. humans |
| --- | :---: |
| g(·) (this package) | **0.910** |
| LLM-judge ensemble (DeepSeek-R1 / Gemini / GPT-4o) | 0.881 |
| Best Bradley-Terry reward model | 0.876 |

See Appendix B for the full benchmark and the *distillation ceiling*
explanation.

## Re-fitting the estimator (optional)

`oral_level_model.pkl` is the version used in every paper number. If you
want to refit on your own data:

```bash
# Pseudo-code; see ideadensity/idea_density_rater_rules.py for the fit logic
python -m orality.fit_estimator  --train_file data/train.json  --out orality/oral_level_model.pkl
```

The bundled pickle stores `{ewm_weights, pca, kmeans, cluster2level}`; all
downstream methods (CAS, LAS, evaluator) load it transparently.
