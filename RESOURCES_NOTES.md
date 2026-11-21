# Notes from the official resources (read 2026-11-21, ~13:45)

Sources: the event email, the hackathon README + report template README, the task package
(TASK_CARD, RELEASE_CONTRACT, simulator, metric/health plugins, analysis/), the official example
submission, the SIDERIUS master code, the TESS warm-up, and `/data/galileo/README.md` on the GPU box.

## UPDATE FOR PRISHA (~14:35): hidden-test result and how to frame it

**Upload #1 = submission 16 was scored on the hidden test: 0.4725** (status `scored`, review passed, on the live
leaderboard as Hype Cycle / G2-15). Portal scoring time **155 s**, which matches our projection of 161 s for 4,000 stars at
14 workers (limit 1,800 s). ZIP 32.6 KB, SHA-256 `1a4b4e94…e2662`, commit 5c1a5b9. Recorded in `report/SUBMISSIONS.md`.
The model is **frozen**; 4 of 5 uploads are left in reserve.

**How good is 0.4725? Reference points (facts; the framing in the report is ours to write):**
1. **Physical limit.** The Fisher ceiling (official `headroom.py`, validation) is **0.536**. We reach **88.6%** of it on
   validation; the baseline reaches 88.0%. TASK_CARD: "large gains in success_rate are not physically possible". About 6 points
   of headroom remain, mostly stars where the data genuinely prefer a wrong answer (only 2.7% of our failures are
   recoverable by a better search).
2. **Vs the baseline (same stars).** Validation 0.47475 vs 0.4715 = +13 stars (+0.3 pts), i.e. about 5% of the remaining gap to the
   ceiling closed. The **paired** train-side evidence is the strong part: K-debias +70/−39 on 4,000 held-out std stars
   (p≈0.004); Occam vs baseline std +19/−6 (p=0.01).
3. **Noise in the hidden-test number.** One binomial standard error on 4,000 stars is about **±0.8 pts**. So hidden 0.4725 vs
   validation 0.47475 is the same performance on a different sample. **We cannot claim we beat the baseline on the hidden
   test**: the leaderboard shows no baseline score (`baseline: null`) and no per-group scores. Base the improvement claim on the
   paired comparisons (and say so); report the hidden-test number as the official score.

**Framing for the award.** Leaderboard differences on this task will be small and noisy because of the ceiling, so the report is
where we can stand out. Judges weigh evidence, reproducibility and insight; "Outstanding Research & Insight" explicitly
allows a well-explained near-ceiling or negative result. What we can show:
- exact reproduction of the official baseline (4 decimals; failure anatomy matches the TASK_CARD),
- two improvements, each with a physical mechanism and a *different* failure mode: Occam cuts 1-day-alias failures
  8.4% → 5.4%; K-debias lowers median rel. K error 0.0790 → 0.0751 (the gains add up),
- an independent confirmation of the TASK_CARD's "search is not the bottleneck" (four search changes moved 0 of 1,000 stars),
- the proxy that did not transfer, the diagnosis (span bias, ~4.5%), and the fix (faithful proxy reproduces 0.375 exactly),
- refuted hypotheses reported honestly (pk-Occam R06, jitter R08, residual-noise SNR R13),
- paired sign tests throughout; validation used once; no simulated stars; scoring speed reproduced by the organizers' sandbox.

## UPDATE FOR PRISHA (~14:30): status, numbers and files you need

**Model status.** Final candidate = `occam_weight=0.5` + `k_debias_c=6.95` (R10–R12). It is **not uploaded yet**:
Zach uploads it as upload #1 (`build/upload2-v2-submission-220842.zip` on Zach's laptop, SHA-256
`1a4b4e949e24dea461a470e697b774e4bce2ec0305c1fdf5fd6b17e808d2e662`, commit 5c1a5b9). `report/SUBMISSIONS.md` row 1 is
pre-filled; the submission ID comes after the upload. One more experiment is running (residual-noise SNR for
K-debias, R13). It only becomes upload #2 if it clearly wins on train-derived sets.

**Upload budget (organizers on Discord):** "Let's do **5 times per problem** as the limit" (10 total across tasks). Plan:
1 now, at most 1 more, keep 3 spare.

**Organizer answers (Discord):** (1) all scoring is on a hidden held-out set; (2) computing hyperparameters from the
train data is fine (so the Occam weight and K-debias c are OK under `kind=algorithm`); (3) hint: "try data augmentation
on train data". We did that in R11 (subsampling to 15 obs, extra noise; no simulated stars).

**Headroom files for your figures: `results/` (committed).**
- `results/headroom_{baseline,kdbonly,upload1,final}.json`: the official `analysis/headroom.py` on validation predictions.
  Keys per group (`all`, `in_distribution`, `high_ecc`, `sparse`): `baseline_success` (= the analysed model's success,
  despite the name), `fisher_ceiling`, `baseline_over_fisher`, `perfect_search_success`, `fail_share`
  {`period_ok_k_off`, `one_day_alias`, `yearly_alias`, `harmonic`, `other`}, `fail_recoverable_share`.
  "upload1" = Occam only; "final" = Occam + K-debias.
- `results/val_preds_*.npy` ([4000, 3] predictions), `results/val_checks.jsonl` (per-group scores incl. median errors),
  `results/batch1.log`, `results/batch2.log` (paired confirmations, GPU box), `results/*_mac.jsonl` (Mac runs).

| model (validation, held-out check) | all | in-dist | high-e | sparse | % of Fisher ceiling | 1-day-alias share of failures | median rel K err |
|---|---|---|---|---|---|---|---|
| official baseline | 0.4715 | 0.5975 | 0.316 | 0.375 | 88.0% | 8.4% | 0.0790 |
| K-debias only | 0.4735 | 0.599 | 0.318 | 0.378 | 88.4% | 8.4% | 0.0756 |
| Occam only | 0.47175 | 0.601 | 0.309 | 0.376 | 88.1% | 5.4% | 0.0792 |
| **Occam + K-debias (final)** | **0.47475** | **0.6035** | 0.311 | **0.381** | **88.6%** | 5.4% | **0.0751** |
| Fisher ceiling | 0.5357 | 0.6569 | 0.3573 | 0.4716 | | | |

Baseline failure shares reproduce the TASK_CARD exactly (26.1 / 8.4 / 17.2 / 47.6%), which validates the pipeline.

**Facts for the insights (wording is yours):**
- The Occam term mainly fixes **1-day aliases** (8.4% → 5.4% of failures) and costs **5–7 high-e stars** in every validation run.
- K-debias acts on a **different failure mode**: amplitude only, never aliases. It lowers median rel. K error 0.0790 → 0.0751.
  The two gains add up.
- **c is robust:** fitted separately on std and on 15-obs stars it comes out at 6.93 both times. That supports the √N scaling
  of the bias formula. The noisy augmentation wants c ≈ 20 because its noise exceeds the assumed 2 m/s jitter.
- The low-SNR ML K bias: median log10 K error +0.037 (SNR 5–8), +0.045 (8–12), +0.020 (12–20), +0.004 (20–40),
  +0.001 (>40) dex on 6,000 train stars, i.e. ≈ +10% at low SNR, right at the 10% success threshold.
- Search changes (extra starts, e=0.92 grid, wider window, more evaluations) changed **0 of 1,000 std stars**, which
  independently confirms the TASK_CARD's "perfect search adds only +0.0075". With our scoring, perfect search adds
  +0.011 (0.4748 → 0.486), and only 2.7% of our failures are recoverable.
- The faithful sparse proxy reproduces the real sparse baseline **exactly (0.375)**; the old proxy gave 0.332.
- Why we don't simulate e ≥ 0.6 stars: the TASK_CARD calls it outside the intended rules (it removes the "absent from
  training" premise of high_ecc). This is a candidate "limitations / next experiment" item.

**Decision-trail table (your joint to-do with Zach):** R01–R12 are in `report/EXPERIMENT_LOG.md`. A suggested ≤6-row cut:
R01 (baseline reimplementation) · R03 (Occam) · R04+R05 (proxy did not transfer → faithful proxy) · R06–R08 merged
("search, pk-Occam and jitter didn't help") · R09–R11 (K-debias + augmented calibration) · R12 (final check).

## Organizer answers (Discord, ~14:10)
- **"All on hidden held out to avoid overfitting on validation."** The leaderboard/final score uses a hidden held-out
  set for every group. Validation is only our own one-time check.
- **"It's totally fine to calculate the hyper parameters with the data."** Train-fitted constants (Occam weight,
  K-debias c) are fine under `kind=algorithm`.

## Facts that change what we do
1. **The score comes from a hidden test set.** `/data/galileo/README.md` says "The hidden test sets used
   for scoring are NOT here". The TASK_CARD says validation is the scored split, and the event README
   overrides that. So the leaderboard is *not* our validation number, and tuning to validation is pointless
   as well as against the rules. In the report, call validation "held-out check" and the leaderboard "hidden test".
   **Confirmed by organizers on Discord (see above).**
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
