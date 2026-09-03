# Design Decisions and Trade-offs

Written for the interview question *"why did you build it this way?"*.

## Validation: walk-forward only

Random `train_test_split` on a time series leaks the future into the past through lag features
and gives an accuracy number that will never be reproduced in production. Every fold here trains
on `date <= cutoff` and tests on the next block. `assert_no_leakage` is asserted in the test suite,
not just documented.

**Trade-off:** fewer effective training rows in the early folds and higher variance across folds.
That variance is reported (`rmse_std`) rather than hidden.

## One global model over per-series models

A single model across all `(store, product)` series shares seasonality and promotion effects,
handles short series, and keeps serving simple. Series identity enters as ordinal codes plus
per-series lag/rolling features.

**Trade-off:** series with genuinely different dynamics are averaged together. The
coefficient-of-variation table on the EDA page flags where that is likely to hurt, and Prophet
(fit per series) is available as a comparison.

## Recursive forecasting rather than direct multi-horizon models

Lag features make one-step-ahead prediction accurate. For horizon *h*, predictions are fed back
in as lags. That keeps one model for every horizon.

**Trade-off:** errors compound with horizon. The alternative — one model per horizon — multiplies
training cost and loses the shared signal. `window_days` bounds how much history is rebuilt per
step; the expanding-mean feature is then approximate, which is an accepted, documented cost.

## Rolling statistics computed on the shifted series

`groupby.shift(1)` before `.rolling(w)` rather than `.rolling(w).shift(1)`. Both avoid leakage;
the first is unambiguous when a series has gaps. There is a dedicated test that injects a spike
and asserts the spike's own row does not see it.

## Fourier terms on a fixed epoch

Seasonal terms are computed from a constant origin (2000-01-01), not from the frame's minimum
date. Otherwise a windowed rebuild during recursive forecasting would silently shift the phase
of every seasonal feature relative to training.

## Naive baseline is a first-class model

The seasonal-naive model is trained and scored like everything else, and every model reports a
skill score against it. A gradient-booster that cannot beat "same day last week" is not a result.

## Ordinal encoding over one-hot

Tree models split on ordinal codes without the dimensionality blow-up of one-hot on high-cardinality
store/product identifiers. The encoder is fitted on training data only and maps unseen categories
to `-1`, so a new store at inference time degrades instead of crashing.

**Trade-off:** the codes imply an order the model may exploit spuriously. Acceptable for trees;
it would not be for a linear model.

## MAPE reported alongside RMSE, not instead of it

Retail panels contain zero-sales days. MAPE is undefined there, so it is computed over non-zero
actuals only and sMAPE is reported next to it. RMSE stays the primary selection metric because it
matches the cost of large stock-outs.

## Config-driven, not argument-driven

Paths, feature settings, split strategy, model list and hyperparameters live in
`config/config.yaml` and `config/model_params.yaml`. Reproducing a run means keeping one file,
and the dashboard can override values per session without touching code.

## Model bundle stores the feature contract

`ModelBundle` serialises the model *and* the fitted `FeatureBuilder` and its feature-name list.
Inference rebuilds features with the same object that trained them, which removes the most common
class of production bug: train/serve feature skew.

## Rule-based insight engine rather than an LLM

Every recommendation carries the evidence that produced it (`uplift_pct`, `change_pct`, store id).
That is auditable and testable. The cost is coverage — the rules only see what they were written
to see.

## Known limitations

- No prediction intervals yet; the forecast is a point estimate.
- Recursive forecasting assumes future promotions/holidays are known or set via a scenario.
- Prophet is capped at `max_series` fitted series to bound training cost on wide panels.
- The insight thresholds (8%, 10%, CV 0.6) are heuristics, not calibrated to a specific retailer.
