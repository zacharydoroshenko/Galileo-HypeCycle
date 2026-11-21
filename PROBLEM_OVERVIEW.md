# Problem Overview (for ML folks, no physics needed)

## Input data

Each sample is one star, stored as a small array of shape `[4, 60]`. Think of it as an **irregularly sampled, variable-length time series** padded to 60 steps:

| Channel | Meaning | ML analogy |
|---|---|---|
| 0: time | When each measurement was taken (days) | Timestamps, **unevenly spaced** with big gaps |
| 1: radial velocity | How fast the star moves toward or away from us (m/s) | The signal value |
| 2: error | Reported noise level of each measurement | Per-point label noise / uncertainty |
| 3: mask | 1 = real observation, 0 = padding | Attention / padding mask |

Each star has a few dozen real points; the rest is padding. The channel order is our assumption until we read `TASK_CARD.md`.

There are no engineered features, just raw sequences. The data has some features that make it different from typical time-series work:
- **Gaps follow a pattern.** Observations happen only at night and only in certain months each year, so there are regular daily and yearly holes. That creates fake periodicities called aliases.
- **There's hidden extra noise.** Each star has extra random noise ("jitter") that isn't in the error channel. The reported uncertainties are underestimates.

## What we predict

The signal is a **periodic wave** caused by an orbiting planet pulling the star back and forth. We predict three numbers that describe that wave:

| Output | Physical meaning | Signal-processing meaning |
|---|---|---|
| `log10 P` | Orbital period in days | How often the wave repeats |
| `log10 K` | Wobble amplitude in m/s | How tall the wave is |
| `e` | Eccentricity (0–0.95) | How lopsided the wave is: 0 = clean sine, high = a sharp spike plus a long flat stretch |

So it's a **3-output regression**, but the outputs are really parameters of a known equation (Kepler's laws). The true wave has three more nuisance parameters we fit but don't report: phase, orientation and a constant offset.

## Success metric

`success_rate` = the fraction of stars where **both** of these hold:
- the predicted period is within **±1%** of the truth
- the predicted amplitude is within **±10%** of the truth

Eccentricity is not scored.

This works like **classification accuracy, not MSE**. Each star is a hit or a miss, with no credit for being close. The period requirement is very strict, and the usual failure is landing on an alias: half, double, or a "daily/yearly shifted" version of the true period. That's a 50–100% error, not a small one.

The test set has three slices: stars like the training ones, stars with **higher eccentricity** than anything in training (out of distribution), and stars with **far fewer observations**.

## Threshold to beat

We don't know the score yet. The package will include a baseline (probably a plain Lomb-Scargle periodogram, which is the classical method), and its score is our first number to beat. After that, the target is whatever leads the leaderboard. Neither number is published until the data is released.

## Why this shapes our approach

There's little to learn across stars. Each star is its own small curve-fitting problem: search for the period, then fit the known equation. So the skeleton is an algorithm, not a neural net, and "training" just means tuning search settings on the train split. A neural net would struggle to reach 1% period precision from about 30 irregular points, especially on the out-of-distribution eccentric slice. Physics-based fitting extrapolates to that slice for free.
