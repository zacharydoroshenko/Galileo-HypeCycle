# Submission record: exoplanet-orbits (team G2-15)

Do **not** put the portal access code, API keys or the GPU IP here; this file is committed to git.
Checksum: `shasum -a 256 <zip>`. The report must cite the final row's submission ID, SHA-256 and commit.

| # | Time uploaded | Submission ID | ZIP filename | SHA-256 | Git commit | Config | Status | Hidden-test score (overall / in-dist / high-e / sparse) |
|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | `occam_weight=0.5` | | |
| 2 | | | | | | | | |
| 3 | | | | | | | | |

**Final model for the report:** # ___   **Report PDF uploaded:** ___ (linked to submission ID ___)

## Pre-upload checklist (Prisha)
- [ ] `python -m json.tool submission.json` passes; `kind: algorithm`, `seed`, `initial_artifact`, `plugin`, `train_command`, `train_dir` all present
- [ ] `unzip -l` shows `submission.json` at the root (not in a subfolder)
- [ ] every path named in the manifest exists in the ZIP
- [ ] `train_command` run in a fresh folder with only the train split writes the final plugin
- [ ] plugin smoke test: finite output, shape [N,3], 0 <= e < 1, std(log10 P) >= 0.05
- [ ] no secrets, `EMAIL.md`, logs, `__pycache__`, `.DS_Store` or symlinks; ≤100 MiB zipped, ≤2,000 files
- [ ] ≥3 min since the last upload to this task
