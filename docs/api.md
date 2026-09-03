# API Reference

Base URL: `http://localhost:8000` — interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness, version, whether a model is loaded |
| GET | `/` | Service banner |
| POST | `/data/upload` | Upload CSV/Parquet, returns a validation report and a `dataset_id` |
| GET | `/data/presets` | Available schema presets |
| POST | `/forecast` | Multi-horizon forecast, optional scenario overrides |
| POST | `/insights` | Ranked business recommendations |
| GET | `/model/metadata` | Model name, target, feature contract, CV metrics |
| POST | `/model/reload` | Reload the newest bundle from `models/` |
| GET | `/schema` | Canonical schema and column descriptions |
| GET | `/predictions/history` | Recent requests served by this instance |

## `POST /forecast`

```json
{
  "horizon": 30,
  "dataset_id": null,
  "store": null,
  "product": null,
  "overrides": {"promotion": 1},
  "aggregate": false
}
```

- `dataset_id` — from a prior upload; omitted means the bundled sample.
- `store` / `product` — restrict the forecast to one series.
- `overrides` — scenario levers: `promotion`, `holiday`, `marketing_spend`, `inventory`, `temperature`.
- `aggregate` — collapse to one row per date.

Response carries `request_id`, `model_name`, `summary` (totals and change vs the recent window)
and the forecast `records`.

## Status codes

| Code | Meaning |
|---|---|
| 400 | Unknown schema preset |
| 404 | Unknown `dataset_id`, or no rows for the requested store/product |
| 413 | Upload above `api.max_upload_mb` |
| 415 | Unsupported file type |
| 422 | Body failed validation (for example `horizon` out of range) |
| 503 | No trained model available — run the pipeline first |
