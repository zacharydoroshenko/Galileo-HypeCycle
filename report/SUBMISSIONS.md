# Submission record: exoplanet-orbits (team G2-15)

Upload budget (updated rules): **5 uploads total**. **Used: 1 of 5** (row 1); 4 left. Nothing was uploaded before row 1; the earlier Occam-only zip was never uploaded.

Do **not** put the portal access code, API keys or the GPU IP here; this file is committed to git.
Checksum: `shasum -a 256 <zip>`. The report must cite the final row's submission ID, SHA-256 and commit.

| # | Time uploaded | Submission ID | ZIP filename | SHA-256 | Git commit | Config | Status | Hidden-test score (overall / in-dist / high-e / sparse) |
|---|---|---|---|---|---|---|---|---|
| 1 | 14:25:38 PST (22:25:38Z) | **16** | `upload2-v2-submission-220842.zip` (local name; built on GPU box as `v2-submission-220842.zip`) | `1a4b4e949e24dea461a470e697b774e4bce2ec0305c1fdf5fd6b17e808d2e662` | 5c1a5b9 | `occam_weight=0.5`, `k_debias_c=6.95` (validation check: 0.47475 / 0.6035 / 0.311 / 0.381) | **scored** (14:29, review passed, on leaderboard; portal: scoring time 155 s, ZIP 32.6 KB; our projection was 161 s for 4,000 stars at 14 workers) | **0.4725** / per-group not shown on leaderboard |
| 2 | | | | | | | | |
| 3 | | | | | | | | |
| 4 | | | | | | | | |
| 5 | | | | | | | | |

**Final model for the report:** # 1 (submission ID 16). Late experiments R14–R17 found no replicated improvement, so no further uploads (4 unused).   **Report PDF uploaded:** ___ (linked to submission ID ___)

## Pre-upload checklist (Prisha)
- [ ] `python -m json.tool submission.json` passes; `kind: algorithm`, `seed`, `initial_artifact`, `plugin`, `train_command`, `train_dir` all present
- [ ] `unzip -l` shows `submission.json` at the root (not in a subfolder)
- [ ] every path named in the manifest exists in the ZIP
- [ ] `train_command` run in a fresh folder with only the train split writes the final plugin
- [ ] plugin smoke test: finite output, shape [N,3], 0 <= e < 1, std(log10 P) >= 0.05
- [ ] no secrets, `EMAIL.md`, logs, `__pycache__`, `.DS_Store` or symlinks; ≤100 MiB zipped, ≤2,000 files
- [ ] ≥3 min since the last upload to this task
