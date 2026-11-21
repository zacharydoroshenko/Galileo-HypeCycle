# Notes from the official resources (read 2026-11-21, ~13:45)

Sources: the event email, the hackathon README + report template README, the task package
(TASK_CARD, RELEASE_CONTRACT, simulator, metric/health plugins, analysis/), the official example
submission, the SIDERIUS master code, the TESS warm-up, and `/data/galileo/README.md` on the GPU box.

## Facts that change what we do
1. **The score comes from a hidden test set.** `/data/galileo/README.md` says "The hidden test sets used
   for scoring are NOT here". The TASK_CARD says validation is the scored split, and the event README
   overrides that. So the leaderboard is *not* our validation number, and tuning to validation is pointless
   as well as against the rules. In the report, call validation "held-out check" and the leaderboard "hidden test".
   (To confirm on Discord if we want to quote it.)
2. **10 uploads per team** (the email and RELEASE_CONTRACT both say 10; the "5" in the guide is outdated).
   A timely accepted upload can stay queued and be scored after 4:00 pm.
3. **The simulator is public** (`package_exo/simulator/generate_rv_data.py`). The rules say to learn only from the
   train split, and that simulating extra stars (e.g. e >= 0.6) is outside the intended rules and must be disclosed.
   We do **not** simulate stars. Our proxies only subsample observations of existing train stars.
4. **Validation parameter tables are on the GPU box** (`/data/galileo/exoplanet-orbits/data/rv_validation_params.npy`,
   with true jitter, omega and phase). The official `package_exo/analysis/headroom.py` uses them to compute the
   per-star **Fisher ceiling**, the **perfect-search ceiling** and the **official failure classes** for *any*
   predictions file. This gives the "success vs theoretical best" chart (see the Prisha section below).
5. **Our metric code matches the official one exactly.** The `plain` rv_v2 config reproduces the published baseline
   to 4 decimals on validation (0.4715 / 0.5975 / 0.316 / 0.375). Tolerances are strict `<`.
6. **Health gates** (all blocking): finite outputs, `-1e-6 <= e < 1`, P and K finite and positive, and a std of
   predicted log10 P of at least 0.05 dex. We clip e to [0, 0.95].
7. **Plugin loading.** SIDERIUS loads plugins with `importlib.spec_from_file_location` and registers them in
   `sys.modules` under a prefixed name. Our pool handles both the registered and the unregistered case. The
   Linux fork path was verified on the box, with a projected 161 s for 4,000 stars at 14 workers (limit 1,800 s).

## Proxy-set finding (important for the report)
- Upload #1 (Occam 0.5) gave +1.2 points (std) and +1.8 points (sparse) on the **train proxies**, but
  only +0.35 (in-dist), +0.1 (sparse) and -0.7 (high_ecc) on **validation**.
- The simulator picks one *random* night in the first and last seasons. Our old sparse proxy kept the
  *earliest/latest* of 30-60 observations, so its time spans were ~4.5% too long (mean 1,064 d vs 1,018 d for
  the real sparse stars). `tools/make_sparse_v2.py` builds a simulator-faithful proxy (mean 1,024 d).
- Use **means**, not medians, for spans. The span distribution is multimodal (2-5 yearly seasons), and its
  median jumps between ~950 and ~1,160 d with tiny changes in the season mix.
- The high-e proxy (train e up to 0.6) cannot represent high_ecc (e 0.6-0.9). Occam's high_ecc loss is
  plausibly because the full-Jacobian Occam term penalises sharp eccentric fits; `occam_mode="pk"`
  (marginal (lnP, lnK) covariance only) tests that hypothesis.

## Resources that are not relevant to us
- **SIDERIUS** (agents/orchestration, Luna) is optional. Nothing in the rules requires it or gives credit for it.
  The `n_workers=14` algorithm runner used by the scorer is on the portal side, not in the public SIDERIUS repo.
- **TESS + Luna warm-up**: a practice exercise on a different problem (stellar rotation from light curves,
  training-mean baseline). Nothing in it transfers to this task.
- The OpenAI key is stored on the box in `~/.openai_key` (chmod 600) and exported in interactive shells via `~/.bashrc`.

## For Prisha: report rules (template README + hackathon README)
- **4 pages max, including** title, abstract, figures, tables and the AI-use statement. References may run longer;
  no narrative appendices. Don't change fonts or margins. Compile with `latexmk -pdf main.tex`.
- **Six required sections:** (1) scientific goal and initial hypothesis, with the observation that would support or
  refute it; (2) research trajectory as **Run ID / why / change / result / next decision**, real negative results
  included, with full logs linked externally (our GitHub repo); (3) final model and comparisons: **official baseline vs
  final vs a meaningful alternative**, with split, metric, gates, compute/time and uncertainty; (4) up to 3 key insights
  as claim / evidence / limitation; (5) limitations and next experiment; (6) reproducibility, contributions, AI use.
- **Bind the report to the exact submission ID + zip sha256 + git commit.** Fill in Team ID `G2-15` and task slug
  `exoplanet-orbits` in the title.
- **AI may organise, translate and proofread, but must NOT generate the core arguments.** Judgments, insights,
  limitations and loophole explanations must be written by us. Disclose AI coding assistance and how we verified it.
- "Describe actual data use accurately; do not imply held-out generalization" that we didn't measure: the proxies are
  train-derived, and validation was used only as a one-time check.
- Awards: Scientific Solution x2, Research & Insight x1 (**a well-supported negative result can win**), Reward Hacking x1.

## For Prisha: commands on the GPU box
```sh
ssh galileo                     # host alias configured on Zach's laptop; others: ssh -i <key> team@<IP from email>
source /opt/siderius/.venv/bin/activate
cd ~/work/hype                  # our code (synced by rsync; not a git checkout)
# Fisher ceiling + official failure classes for any validation predictions (.npy [4000,3] or predictions.json):
python ~/work/exoplanet-orbits/package/rv-exoplanet/analysis/headroom.py \
    --zenodo_dir /data/galileo/exoplanet-orbits/data --predictions <preds.npy> --split validation \
    --workers 14 --output build/headroom_<name>.json
```
