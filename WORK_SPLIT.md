# Work Split: exoplanet-orbits (Track 2, Physics)

**Zach** owns the model and all uploads. **Prisha** owns the report and the submission record.
Only Zach uploads models, so the upload budget and the 3-minute gap between uploads stay under one person's control.
The extension, the new deadlines and the hidden-test scoring have been confirmed with the organizers on Discord.

## Ownership

Each column is that person's to-do list, in order (1 → 4).

| Area | Zach | Prisha |
|---|---|---|
| Model (`solution/rv_v2.py`) | **1.** Finish the remaining batch-2 configs (R10) and decide on K-debias. Upload #2 candidate: `occam_weight=0.5` + `k_debias_c=6.95` (already set in `train_v2.py`). Already tested and dropped: yearly/peaks5/bounds (R02), `occam_mode="pk"` (R06), search changes (R07), jitter/jitter_dof (R08). Optional if time allows: a train-label P/K prior for the sparse regime. **3.** Freeze the final model | — |
| Ablations and confirmation | Run `tools/ablate_v2.py` / `tools/confirm_v2.py` on held-out TRAIN stars | — |
| Theoretical-best numbers and figures | **2. Before 4:00 (GPU access ends):** run `headroom.py` on the GPU box on the validation predictions of the baseline and the final model (`score_val --save/--set` writes them); copy the JSON files to the laptop | **1.** Get the `headroom.py` JSON files from Zach before 4:00. Write the plotting script and save `report/figs/success_vs_ceiling.pdf` and `report/figs/failure_breakdown.pdf` |
| Experiment log | Add each run to `report/EXPERIMENT_LOG.md` (what changed, why, result, decision) | — |
| Packaging and uploads | Build the ZIP with `tools/package_v2.py`; upload a safe first model early, then the final one; leave ≥3 min between uploads; send Prisha the submission ID and SHA-256 after each upload | **2.** Before each upload, go through the pre-upload checklist in `report/SUBMISSIONS.md` with Zach. After each upload, record the submission ID, SHA-256, git commit and score there |
| Report (`report/main.tex`, ≤4 pages) | **4.** Fill in the ZACH placeholders (`grep -n 'fillin{ZACH' report/main.tex`): the method paragraph, the why/decision cells of the decision trail (in his own words) and his contributions. With Prisha, cut the decision trail to ≤6 rows (e.g. merge R06–R08 into one "didn't help" row; R08 is not in the table yet) | **3.** Write §1 (goal and hypothesis), Insight 2 (the proxy-set finding: Occam's gain on train-derived sets shrank on validation, and the old sparse proxy covered spans ~4.5% too long; see R04 and R05), §5 (limitations) and §6 (reproducibility, contributions, AI use). Throughout, call validation the "held-out check" and the leaderboard the "hidden test", and keep the statement that we simulated no stars |
| Final report | Review the full draft after the model freeze | **4.** At model freeze: put in the final numbers and the binding details (submission ID, SHA-256, commit); write the abstract and title last; check the body fits in 4 pages and the PDF is under 10 MiB; upload the PDF at https://submit.siderius.scitra.org/track2/reports, linked to the final submission ID |

Both of us write the report's core arguments ourselves (AI may only organize and proofread).

## Sync points

| When | What |
|---|---|
| Each upload | Prisha's checklist passed → Zach uploads → submission ID, SHA-256 and commit logged in `report/SUBMISSIONS.md` |
| ~3:30 | Zach has run `headroom.py` and copied the JSON files to the laptop; Prisha builds the figures |
| Model freeze | Final model chosen and uploaded; numbers final |
| After freeze | Both edit the report; Prisha uploads the PDF before the report deadline |

## Rules to keep in mind
- Tune only on `data/exo/train/`. `val/` is only for a one-time check (`tools/score_val.py`), never for tuning.
- No new pip installs; numpy and scipy only in the plugin; `kind: "algorithm"`, with no weight files.
- 10 accepted uploads per team; an upload that fails evaluation still counts.
- No access code, API key or GPU IP in any committed file, the report or the ZIP.
