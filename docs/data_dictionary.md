# Data Dictionary

## Canonical schema

Every dataset is mapped onto these columns before anything else runs.

| Column | Type | Required | Missing-value policy | Description |
|---|---|---|---|---|
| `date` | datetime | yes | rows with unparseable dates are dropped | Observation date |
| `store` | categorical | yes | blank values are a validation error | Store / outlet identifier |
| `product` | categorical | yes | blank values are a validation error | Product, SKU or family |
| `sales` | numeric | yes | `cleaning.fill_missing_target` (drop / zero / interpolate) | Forecast target |
| `promotion` | binary | no | filled with 0 | On promotion that day |
| `holiday` | binary | no | filled with 0 | Public or company holiday |
| `region` | categorical | no | filled with `unknown` | Geographic grouping |
| `category` | categorical | no | filled with `unknown` | Product grouping |
| `temperature` | numeric | no | forward/backward fill, then median | Daily temperature |
| `inventory` | numeric | no | forward/backward fill, then median | Stock on hand or transaction count |
| `marketing_spend` | numeric | no | forward/backward fill, then median | Marketing or markdown spend |

Non-numeric event columns (for example Rossmann's `StateHoliday='a'`) are coerced to
`1` when they carry any non-empty, non-zero value.

## Schema presets

| Preset | Source dataset | Notable mappings |
|---|---|---|
| `canonical` | already-canonical files | identity |
| `favorita` | Kaggle *Store Sales – Time Series Forecasting* | `store_nbr→store`, `family→product`, `onpromotion→promotion`, `transactions→inventory` |
| `rossmann` | Kaggle *Rossmann Store Sales* | `Promo→promotion`, `StateHoliday→holiday`, constant `product="ALL"` |
| `walmart` | Kaggle *Walmart Store Sales* | `Dept→product`, `Weekly_Sales→sales`, `MarkDown1→marketing_spend` |

## Engineered features

| Group | Examples | Transformation |
|---|---|---|
| Calendar | `dayofweek`, `weekofyear`, `is_month_end`, `is_payday_window` | Derived from `date` |
| Fourier | `sin_7_1`, `cos_365_25_3` | Seasonal basis on a fixed epoch (2000-01-01) so a windowed rebuild keeps the same phase |
| Events | `promotion_lead_1`, `holiday_lag_1`, `promotion_roll_7` | Shifted per series |
| Lags | `sales_lag_1/7/14/28` | `groupby(store, product).shift(lag)` |
| Rolling | `sales_roll7_mean`, `sales_roll28_std` | Computed on the **shifted** series, so row *t* never sees *y_t* |
| Expanding | `sales_expanding_mean` | Shifted expanding mean |
| Momentum | `sales_momentum_7` | `lag_1 - lag_7` |
| Interactions | `promo_x_recent_mean`, `holiday_x_weekend` | Products of drivers and recent level |
| Identifiers | `store_code`, `product_code`, `region_code`, `category_code` | Ordinal encoding fitted on train only; unseen values map to `-1` |

Rows whose lag features are still undefined (the warm-up window) are dropped before training.

## Validation rules

| Rule | Severity | Trigger |
|---|---|---|
| `required_columns` | error | any of date/store/product/sales missing |
| `min_rows` | error | fewer rows than `validation.min_rows` |
| `invalid_dates` | error | unparseable dates |
| `missing_target` | error / warning | above / below `max_missing_target_ratio` |
| `target_dtype` | error | target is not numeric |
| `duplicates` | error / warning | duplicate `(date, store, product)` above / below `max_duplicate_ratio` |
| `blank_store`, `blank_product` | error | empty identifiers |
| `negative_sales` | warning | negative target values |
| `date_range`, `future_dates` | warning | dates before `min_date` or beyond today |
| `date_gaps` | warning | series with missing calendar days |
