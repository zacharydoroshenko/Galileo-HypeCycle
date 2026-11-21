# Galileo Hackathon: exoplanet-orbits (Track 2, Physics)

The task: recover the orbit of a single planet (P, K, e) from sparse radial-velocity (RV) data.
- Input: `X [N, 4, 60]`. Output: `[N, 3] = [log10 P, log10 K, e]`.
- Metric: `success_rate`, the share of stars with P within 1% **and** K within 10%. e is **not scored**.
- Test regimes: in-distribution, more eccentric than any training star, and far fewer observations.
- Submission kind: `algorithm`, a CPU plugin with `predict(X, n_workers)` (14 workers given). Scoring has a 1800 s limit and retraining has a 3600 s limit.
- Full rules: `docs-track2-guide.md`. Report template: `report/`.

## Layout
| path | what |
|---|---|
| `solution/rv_plugin.py` | **the solver** (self-contained, numpy+scipy only). GLS periodogram → candidates + aliases/harmonics → (P,e,M0) grid with exact linear solve for (K cosω, K sinω, γ) → Nelder-Mead polish with jitter in the likelihood |
| `train/train.py` | "training" = tunes `CONFIG` on the TRAIN split and writes the final plugin with the tuned config baked in. `load_train()` must be adapted to the real file layout |
| `tools/inspect_data.py` | run first on the real data to determine the channel layout, units, and label ranges |
| `tools/synth.py` | synthetic generator (standard / eccentric / sparse) for stress tests |
| `tools/evaluate.py` | local mirror of the metric plus a failure breakdown (harmonic vs other P misses, K misses) |
| `tools/diagnose.py` | per-star: search failure (true orbit fits better but we didn't find it) vs unidentifiable (low SNR) vs ambiguous. **Run on real train first** to decide where effort goes |
| `tools/candidate_recall.py` | share of stars whose true P is in the candidate list (search recall before fitting) |
| `tools/test_prior.py` | A/B of the learned (log P, log K) prior weight |
| `tools/check_parallel.py` | 40 s check: pool == serial, tuned CONFIG reaches workers, no crashes |
| `tools/package.py` | stage, rerun `train_command` in a fresh dir, smoke test, then zip with `submission.json` at the root |

## Synthetic baseline (pre-data, 80 stars each)
| regime | success | P ok | K ok |
|---|---|---|---|
| standard | 0.64 | 0.79 | 0.68 |
| eccentric (e 0.6–0.92) | 0.36 | 0.58 | 0.39 |
| sparse (8–15 obs) | 0.33 | 0.41 | 0.49 |
Speed is about 2 CPU-s per star, so roughly 10k stars fit in the scoring limit with 14 workers.

### Findings since (60–80 synthetic stars each, so differences of ±1–2 stars are noise)
- **Diagnosis:** on standard stars, all misses are unidentifiable (median SNR ~1.7). There are no search failures, so better searching won't help there. On eccentric stars, 15% are search failures.
- **Keplerian periodogram** (`kep_on`, a scan with eccentric templates): candidate recall on eccentric stars goes from 66% to 79% and P-correct from 52% to 65%. Success is about flat (35→37% eccentric, 38→37% sparse, 60→60% standard) because K misses dominate. Cost is about 3×. On by default because it targets the eccentric test set; `train.py` can switch it off.
- **Learned (log P, log K) prior** (`prior_weight`): no effect on synthetic data, which is expected because synth labels are log-uniform. Default 0; `train.py` tunes it on the real labels. Deliberately **no prior on e**, because the test set is more eccentric than training.
- Finer (P, e, phase) grids gave no gain at 2–4× the cost.
- **The remaining big lever is K for eccentric orbits.** Candidates are P-right/K-wrong when no point samples periastron.

### Engineering notes
- `true_anomaly_table` replaces Kepler solves in the grid stages (max error 5e-4 rad), giving a 1.6× speedup. The polish step uses exact solves.
- Pool: `fork` on Linux (sandbox), `spawn` on macOS (fork crashes Accelerate-linked children), and `ProcessPoolExecutor` so dead workers raise instead of hanging. Any failure falls back to serial. CONFIG is shipped with each task.

## Leaderboard
- Page: https://siderius.scitra.org/hackathon/leaderboard?view=projector (JS-rendered, provisional, reviewed scores only)
- Terminal: `python3 tools/leaderboard.py` (polls https://submit.siderius.scitra.org/api/leaderboard, the page's own JSON feed)
- We are **G2-15 "Hype Cycle"**. The leaderboard scores on *validation*, so use it as a check, never as a tuning signal.

## Setup (new teammate: run these three lines)
```sh
uv venv -p 3.12 .venv && uv pip install -p .venv numpy scipy pandas h5py matplotlib scikit-learn threadpoolctl
.venv/bin/python tools/get_data.py      # task package + Zenodo data, checksum-verified (~50 MB, not in git)
.venv/bin/python tools/evaluate.py --data data/exo/train_sub300.npz --n 300 --workers 8
```
`get_data.py` creates `package_exo/` (TASK_CARD.md, baseline, simulator, metric plugins, official example
submission) and `data/exo/{train,val}/` with the official file names. **Tune only on `data/exo/train/`.**
`val/` is for checking only: the task rules forbid tuning on it, and generating extra stars with the
simulator is outside the intended rules.

Official baseline (TASK_CARD): validation success 0.4715 (in-dist 0.5975, high_ecc 0.316, sparse 0.375),
0.24 CPU-s/star. The information ceiling is 0.536, so headroom is small; see TASK_CARD "Difficulty".
On our 300-star train subset the baseline scores 0.593.

## Run sheet for challenge day
**11:45–1:00 (onboarding)**: SSH in, `source /opt/siderius/.venv/bin/activate`, clone this repo onto the GPU box, check that the OpenAI key works.
**At package release (noon):**
1. `cp -r /data/galileo/exoplanet-orbits/package ~/work/exoplanet-orbits/` and run `fetch_data.py --source_dir /data/galileo/exoplanet-orbits/data --output_dir ~/work/exoplanet-orbits/data`
2. Read `TASK_CARD.md` and `baseline/`. Record the **exact plugin contract, manifest fields, channel order, time units, and the official evaluator**.
3. `python tools/inspect_data.py <train dir>`, then fix `CONFIG` (`ch_*`, `time_scale`, `p_min`/`p_max`) and `load_train()`.
4. Retune `tools/synth.py` distributions to match the train labels (P, K, e, n_obs, err ranges).
5. Score the official baseline and ours on a train subset (and on validation for *checking only*, never tuning).

**1:00–~2:15 (improve):** work from the failure breakdown. Likely wins, roughly in order:
- **K misses**: K errors usually come from e/ω degeneracy or jitter. Try a stronger `e_penalty`, more e grid points, and fitting jitter jointly in the grid stage.
- **P misses**: add more peaks, and use the Keplerian likelihood (not GLS power) to pick among aliases. Extend `p_max` if train has P > baseline. Add 1/sidereal-day aliases.
- **Eccentric test set**: GLS is weak for high e, so add a direct (P, e) grid or a "periastron spike" search; keep `e_max` up to 0.95. Don't tune e priors to training (test is *more* eccentric). This is a generalisation point for the report.
- **Sparse test set**: penalise high e / overfitting (BIC-like); use a prior from the train label distribution of P, K (this is legitimate learning from train).
- Optional ML piece: train a small net on train data to **rank candidate periods** (features: GLS power, Δlnlike, P/baseline, n_obs). This stays physics-based but learned.

**~2:15**: first upload (keep a safe one early; uploads have to be ≥3 min apart, the total is 5 or 10 (check the portal), and nothing is accepted after 4:00).
**3:30**: final upload. **4:00–4:30**: report (4 pages): hypotheses, every attempt + why, ablations per regime, failure analysis, limitations, AI-use disclosure.

## Packaging
```sh
python tools/package.py --data_dir ~/work/exoplanet-orbits/data/train   # full tune (~minutes)
python tools/package.py --data_dir ... --quick                          # pipeline check only
# train.py: builds the prior from ALL train labels, then coordinate-wise tunes
# prior_weight, kep_on, e_penalty, n_peaks on --n_tune stars (budget 2400 s)
```
Before uploading, compare `build/submission/submission.json` against TASK_CARD.md. The manifest follows the guide's algorithm form:
`kind: algorithm, seed: 0, initial_artifact, plugin, train_command, train_dir` (no weights).

## Gotchas already handled
- The evaluator may load the plugin via importlib, so multiprocessing pickling breaks. The plugin registers itself and uses a fork pool, with a serial fallback.
- Paths with spaces: env vars are quoted in `train_command`.
- Validity checks: outputs are always finite and per-star; one failing star can't kill the batch.
