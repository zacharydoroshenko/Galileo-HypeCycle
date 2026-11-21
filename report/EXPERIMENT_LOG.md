# Experiment log: exoplanet-orbits (team G2-15)

This is the complete log of our runs, linked from report §2 and §6. Add one row per run, including the ones that failed or were negative.
Proxy sets are built from TRAIN stars only. Validation is used only as a one-time check and never for selection.
Run IDs must match the report's decision-trail table.

| Run ID | Time | Who | Commit | Config / change | Why (hypothesis) | Eval set(s) | Result | Decision |
|---|---|---|---|---|---|---|---|---|
| R01 | ~13:00 | Zach | bf0c99d | `rv_v2`, all switches off (= official baseline rule) | | validation (check) | 0.4715 / 0.5975 / 0.316 / 0.375, matches TASK_CARD to 4 dp | |
| R02 | ~13:10 | Zach | bf0c99d | ablation: plain, jitter, occam, yearly, bounds, peaks5, all | | train_sub300, sparse15, hi_e (300 each) | see `build/ablation.jsonl` | |
| R03 | ~13:15 | Zach | bf0c99d | `occam_weight=0.5` paired vs baseline | | 1,000 std + 600 sparse15 held-out train | std 0.582→0.594 (+17/−5, p=0.017); sparse15 0.330→0.348 (+11/−0, p=0.001) | upload #1 |
| R04 | ~13:40 | Zach | aed5b7f | upload #1 on validation (one-time check) | | validation | in-dist +0.35, sparse +0.1, high_ecc −0.7 pts vs baseline | |
| R05 | | Zach | aed5b7f | simulator-faithful sparse proxy (`make_sparse_v2`) | old proxy spans ~4.5% too long | | | |
| R06 | | Zach | | `occam_mode="pk"` | full-Jacobian Occam may penalise sharp eccentric fits | | | |
| R07 | | | | | | | | |
