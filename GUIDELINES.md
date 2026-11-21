# Guidelines — Galileo Hackathon, Physics Challenge (exoplanet-orbits)

## Constraints

**Time:** competition 1–4:30 pm PST (extended 30 min by the organizers). Models and GPU access close at 4:30 pm, report PDF closes at 5:00 pm.

**Submissions:** 10 accepted uploads total across all tasks for the whole team (the website's track2-guide page says 5 — confirm with organizers and plan for 5). Failed evaluations do not refund an upload, so budget carefully. Uploads to the same task need a 3-minute cooldown.

**Compute:** H100 GPU + CPU on the organizer's machine for the duration. $30 in OpenAI credits. The exoplanet task is physics/CPU-based, so the GPU matters less here.

**Sandbox:** No internet during retraining. `$DATA_DIR` sees only train-split files. Environment is SIDERIUS v1.1.0, Python 3.12, PyTorch 2.10, NumPy, SciPy. No new pip installs.

**Scoring time limits:** 1,800 seconds for inference, 3,600 seconds for retraining.

**Submission package:** ZIP ≤ 100 MB zipped / 1 GB unpacked / 2,000 files. `submission.json` must be at the ZIP root (not in a subfolder). This is an algorithm (no learned weights), so use `"kind": "algorithm"` and omit weight files.

**Rules:** Train only on the train split, never on validation/test data or labels. Do not modify the official evaluator. Predictions must be finite and not all identical, or the submission fails.

**Report:** At most 4 pages of main content (references can go on extra pages), PDF ≤ 10 MiB, due 5:00 pm (extended).

---

## The Physics Challenge — Exoplanet Orbit Recovery

**Goal:** Given sparse radial-velocity (RV) measurements of a star, recover the orbital parameters of its single planet.

**What is RV?** An orbiting planet pulls its star toward and away from us. That wobble shifts the star's spectral lines through the Doppler effect, which shows up as a periodic velocity signal in meters per second. Most exoplanets found before the transit method were found this way.

**Inputs (`X`): shape `[N, 4, 60]`.** N stars, 4 channels (time, radial velocity, measurement error, mask), up to 60 observations per star, zero-padded. A few dozen night-time observations across a few observing seasons.

**Outputs (`Y`): shape `[N, 3]` = `[log₁₀ P, log₁₀ K, e]` per star:**
- **P** = orbital period in days
- **K** = radial-velocity semi-amplitude in m/s (how much the star wobbles)
- **e** = eccentricity (0 = circular orbit, up to ~0.95)

**Scoring (`success_rate`):** A planet is "found" if `|P_pred/P_true − 1| ≤ 1%` **and** `|K_pred/K_true − 1| ≤ 10%`. Score = fraction of stars where both pass. Eccentricity is **not** scored directly, but a wrong e corrupts P and K.

**Why is this hard?**
- **Aliasing:** Ground-based telescopes observe only at night, so the sampling has daily (1 cycle/day) and yearly (1 cycle/year) gaps. A planet with a period near 1 year could look like it has a period of 0.5 or 2 years.
- **Sparse data:** Only a few dozen points to fit ~6 free parameters (P, K, e, ω, M₀, γ).
- **Unknown stellar jitter:** Stars have their own random velocity noise on top of the measurement error. Only the per-measurement error is reported, not the jitter.
- **Eccentricity degeneracy:** Eccentric orbits have sharper, asymmetric RV curves. With few observations, an eccentric orbit can be mistaken for a different period.
- **1% period tolerance is tight.** The Lomb-Scargle periodogram alone rarely lands within 1%, so it needs nonlinear refinement.

**Three test regimes** (all harder than training):
1. Standard stars: like training, but hidden
2. **Eccentric:** eccentricities beyond any training example
3. **Sparse:** far fewer observations than training

---

## What the Skeleton Already Has

The pipeline in `solution/rv_plugin.py`:
1. Parse channels → valid (t, rv, err) arrays
2. Generalized Lomb-Scargle periodogram on a frequency grid
3. Take the top GLS peaks plus their daily/yearly alias frequencies and harmonics as candidates
4. For each candidate: dense grid over (P, e, M₀), with K and γ solved analytically by weighted least squares
5. Nonlinear polish of the top 3 candidates with Nelder-Mead (fitting P, e, M₀ and jitter together)
6. Pick the winner by penalized negative log-likelihood

`tools/synth.py` generates synthetic data before the real data arrives. `tools/evaluate.py` mirrors the official scorer for local benchmarking. `train/train.py` is set up for hyperparameter tuning on the real train split.

---

## Next Steps Once Data Arrives

1. **Run `tools/inspect_data.py`** to verify the channel layout (which index is time, which is RV, etc.). Check the actual period/eccentricity/K distributions against what `synth.py` assumes. The skeleton has `# VERIFY against TASK_CARD.md` comments in several places.
2. **Read `TASK_CARD.md`** in the released package. It defines the exact column order, units (days vs. BJD, m/s vs. km/s) and any quirks.
3. **Run the baseline** that ships with the package before touching anything, and log the score. That's the floor.
4. **Tune `CONFIG` in `rv_plugin.py`** using `train/train.py` on the train split. Key settings: `n_peaks`, `n_polish`, `e_max`, `e_penalty`, `jitter_init`.
5. **Watch the breakdown output** from `evaluate.py`. It sorts failures into "P harmonic miss" (2× or ½× the true period), "K miss" and "other". Harmonic confusion is usually the biggest source of misses.
6. **Package and submit** as `"kind": "algorithm"` (no weights). The skeleton already meets this contract.
