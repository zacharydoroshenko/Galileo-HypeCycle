# Work Split: exoplanet-orbits (Track 2, Physics)

**Zach** owns the model and all uploads. **Prisha** owns the evidence, the reproducibility checks and the report.
Only Zach uploads models, so the upload budget and the 3-minute gap between uploads stay under one person's control.
The deadlines are extended; confirm the new model and report deadlines with the organizers.

## Ownership

| Area | Zach | Prisha |
|---|---|---|
| Model (`solution/rv_v2.py`) | Iterate on the improvement switches: `jitter_fit`, `bound_penalty`, `yearly_aliases` on top of `occam_weight=0.5`; try a train-label prior on P and K for the sparse regime | Optional side experiment on a separate branch (e.g. the sparse-regime prior tuned on the `sparse15` proxy); hand the setting to Zach |
| Ablations and confirmation | Run `tools/ablate_v2.py` / `tools/confirm_v2.py` on held-out TRAIN stars | Re-run the official baseline and the current `rv_v2` setting on the same train-only sets, broken down by regime |
| Failure analysis | Use the results to decide the next change | Use `tools/diagnose.py` and the breakdown from `tools/evaluate.py` to split the misses into K wrong, yearly/daily alias and too little signal; compare with the theoretical best in the TASK_CARD |
| Experiment log | Report each run (what changed, why, result) | Keep the shared log: run ID, commit, settings, reason, result, decision (negative results included) |
| Packaging | Build the ZIP with `tools/package_v2.py` | Before each upload, check that `train_command` works in a fresh folder with the train split only, that `submission.json` is at the ZIP root with all its fields and paths, and that there are no secrets, logs, caches or symlinks |
| Uploads | Upload a safe first model early (baseline + occam), then the final one; leave ≥3 min between uploads | Record each submission ID, ZIP checksum and score |
| Figures and tables | Supply numbers on request | Make the comparison table (official baseline / final model / an alternative, by regime); a chart of success vs. the theoretical best; a failure-breakdown chart; example fits for one star we get right and one where the wrong period wins |
| Report (`report/main.tex`, ≤4 pages) | Write the method and research-trajectory sections; review the full draft | Write goal and hypothesis, comparisons, insights, limitations and next experiment, plus reproducibility; get it compiling early; upload the PDF linked to the final submission ID |
| Reward-hacking section (optional award) | Sanity-check any exploit found | Look into the TASK_CARD's "Known shortcuts and leakage risks"; if we find one, show the gap reproducibly, propose a fix and note when the fix could wrongly flag honest submissions |
| Contributions and AI use | Describe his own contributions and AI use | Write the contributions statement and the AI-use disclosure (tools used, what for, how we checked the output) |

Both of us write the report's core arguments ourselves (AI may only organize and proofread).

## Sync points

| When | What |
|---|---|
| Start | `git pull`; Prisha has the baseline vs. `rv_v2` numbers by regime |
| First upload | Safe model uploaded; Prisha's check passed; submission ID logged |
| Model freeze | Final model chosen and uploaded; numbers final |
| After freeze | Both edit the report; Prisha uploads the PDF before the report deadline |

## Rules to keep in mind
- Tune only on `data/exo/train/`. `val/` is only for a one-time check (`tools/score_val.py`), never for tuning.
- No new pip installs; numpy and scipy only in the plugin; `kind: "algorithm"`, with no weight files.
- Shared upload budget: 10 accepted uploads (the guide says 5; plan for 5). An upload that fails evaluation still counts.
