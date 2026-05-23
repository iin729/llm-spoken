# WSPairs Dataset

**Written-to-Spoken Pairs** — a parallel corpus of 4,709 written / spoken
sentence pairs collected from human–human spontaneous conversations, annotated
with **ordinal Orality Levels 0–5** based on seven interpretable linguistic
features (fillers, interaction markers, stance/hedges, lexical reduction,
repetition, PID, NVR).

## Splits bundled in this repo

| File | # items | Notes |
| --- | --- | --- |
| `train.json` | 3,183 | Main training set used in Tables 1 / 2 of the paper |
| `dev.json` | 471 | Validation set for hyperparameter search |
| `test.json` | 1,055 | Held-out test set used for all reported numbers |
| `calibration.json` | 1,000 | Calibration set for fitting CAS's α = f(t) curve (Sec. 4.1) |
| `examples/examples.json` | 60 | A few inspection examples (Appendix) |

Domain composition (train / dev / test): **MediTOD** (medical triage),
**SpokenWoZ** (task-oriented), **House Sale** (negotiation, from Heddaya et al.
2023). Each split is roughly proportional across the three domains.

## Sample schema

Every JSON item carries both sides of the pair, the orality level, and the
seven sociolinguistic feature counts pre-computed by `orality.stat_all_metrics`:

```jsonc
{
  "source":           "meditod",                // {meditod | spkenwoz | house_sale}
  "written":          "Congratulations. ...",
  "spoken":           "OK, good Congrats. ...",
  "instruction":      "### Instruction ...",    // level-conditioned prompt
  "input":            "",
  "output":           "...",                    // = `spoken`, kept for trainer compatibility
  "type":             "write2spoken",
  "spoken_statics":   {
       "level": 3,                              // <-- 0..5 Orality Level
       "filler": 0.42, "interact": 1.18, "stance": 0.0,
       "simple": 0.0, "repeat": 0.0,
       "PID": 0.65, "NVR": 0.50, "O_score": 1.32
  },
  "written_statics":  { "level": 0, ... }
}
```

## OOD splits (not bundled)

For the three leave-one-domain-out folds (Sec. 5.3, Fig. fig:eff_ood) you can
regenerate the splits from `train.json` / `test.json` by grouping on
`source`:

```bash
python -m data.tools.split_by_source  --in  data/train.json  --out_dir data/ood
```

This produces `data/ood/train_4{meditod,spokenwoz,house_sale}.json` and
matching held-out test files. The `baselines/*/scripts/run_*_ood.sh` scripts
will pick those up automatically once they exist.

## Re-computing Orality Levels

The 7-feature vector and level are produced by the `orality` package. To
recompute them on any new text:

```python
from orality import stat_all_metrics, eval_oscore_and_level
stats = stat_all_metrics("OK so, um, what's going on?")
o_score, level = eval_oscore_and_level(stats)
print(level)   # -> 4
```

`orality/oral_level_model.pkl` bundles the pre-fit EWM weights and the
k-means cluster→level mapping used in the paper.

## License

The annotation layer (orality levels + 7-feature scores) is released under
**MIT** (see top-level LICENSE).

The underlying spoken transcripts come from three source corpora that retain
their own licenses:

- **SpokenWoZ** — Si et al., 2023 (research-only)
- **MediTOD** — Saley et al., 2024 (research-only)
- **Bargaining / House Sale** — Heddaya et al., 2023 (research-only)

Please cite the source corpora alongside this repo when you publish results
on these splits.
