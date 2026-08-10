# Phase 8 transparent service calibration CV (2026-08-10)

## Outcome

The transparent calibration was executed and correctly failed the Runbook
regret Gates. It is not accepted for deployment.

| Evaluation | Top-1 | Median regret | p95 regret | Overhead p95 |
|---|---:|---:|---:|---:|
| uncalibrated 12-cell replay | 33.33% | 33.89% | 101.80% | 0.04% |
| grouped held-out calibration | 41.67% | 13.64% | 54.77% | 0.05% |
| required Gate | report | <=5% | <=10% | <1% |

Only the controller-overhead Gate passes. The machine-readable result is
`docs/Q-TopoMoE_Phase8_service_calibration_cv_20260810.json`; the frozen
uncalibrated baseline is
`docs/Q-TopoMoE_Phase8_uncalibrated_replay_20260810.json`.

## Method and leakage control

The base cost remains measured MoE kernel time, measured mapping communication
cost, route imbalance and migration cost. Kernel invocation accounting uses
40 model layers and, per request, one prefill invocation plus `output_tokens`
decode invocations. Calibration adds only a candidate-specific non-negative
scale and intercept.

Fits use weighted least squares. Bootstrap 95% intervals are converted to an
approximate variance with a 5% median sigma floor so an artificially narrow
five-run interval cannot dominate. W1, W2, W3 and W4 are each held out in turn;
the held-out family's service medians and oracle labels never enter that fold's
fit. Every one of the 12 observations is evaluated exactly once out of family.

The initial normalized-M-histogram implementation was also audited. It counted
only one average kernel invocation and gave 19.31% median / 70.52% p95 regret.
Correct sequence invocation counting improved those figures to 13.64% / 54.77%
but did not make the result acceptable.

## Diagnosis

The residual is structured, not random fit noise. The current repeated screen
has `enable_prefix_caching=True`, but it did not construct or verify controlled
0%, 50% and 100% prefix-hit populations. All requests are closed-loop; Poisson
and burst arrivals are absent. Consequently queueing, batching and implicit
prefix reuse are not identifiable from the current feature set.

The strongest symptom is W4 (32768 input / 128 output): measured e2e p99 for
FP8 falls from 2131.19 ms at C=1 to 524.90 ms at C=4, while the nominal prefix
field is unchanged. Similar candidate-rank reversals occur across families.
A more flexible regression could memorize these 12 cells, but that would not
be a valid selector and is explicitly rejected.

Full-data coefficients are retained in the JSON only for audit and diagnosis.
They are not an accepted deployment manifest.

## Ordered next action

1. Construct requests with measured 0%, 50% and 100% prefix reuse and record
   rendered-token and prefix-identity hashes.
2. Add deterministic closed-loop, Poisson and burst arrival generators and
   record realized inter-arrival statistics.
3. Run the formal crossed matrix on retained candidates, preserving repeated
   service starts and bootstrap uncertainty.
4. Refit the same transparent model with explicit prefix-hit and arrival-mode
   features, then repeat grouped held-out regret Gates. Do not add RL or encode
   per-cell oracle choices.
