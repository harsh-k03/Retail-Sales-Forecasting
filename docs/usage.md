# Usage

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: make install
```

## Run the pipeline

```bash
python scripts/run_pipeline.py                          # bundled sample data
python scripts/run_pipeline.py --models naive lightgbm  # subset of models
python scripts/run_pipeline.py --horizon 90             # longer forecast
python scripts/run_pipeline.py --data data/interim/store_sales_joined.csv --preset favorita
python scripts/run_pipeline.py --tune                   # Optuna search first
```

Outputs land in `reports/` (metrics, figures, validation report) and `models/`.

## Use the real Kaggle data

```bash
pip install kaggle                       # and put your token in ~/.kaggle/kaggle.json
bash scripts/download_kaggle.sh          # -> data/raw/
python scripts/prepare_kaggle.py         # joins stores/transactions/oil/holidays
python scripts/run_pipeline.py --data data/interim/store_sales_joined.csv --preset favorita
```

`--max-rows` on `prepare_kaggle.py` keeps only the most recent N rows if you want a faster loop.

## Dashboard

```bash
streamlit run app/streamlit_app.py       # http://localhost:8501
```

Pages: Home → Upload → Validation → EDA → Features → Models → Forecast → Explainability →
Insights → Scenario Analysis → Settings.

## API

```bash
uvicorn app.api.main:app --reload        # http://localhost:8000/docs
```

```bash
curl -F "file=@data/sample/sample_store_sales.csv" \
     "http://localhost:8000/data/upload?preset=favorita"

curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"horizon": 30, "aggregate": true}'

curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"horizon": 30, "overrides": {"promotion": 1}, "aggregate": true}'

curl http://localhost:8000/model/metadata
curl http://localhost:8000/predictions/history
```

## Docker

```bash
docker compose up --build      # API on :8000, dashboard on :8501
```

## Tests

```bash
pytest                    # everything
pytest -m "not slow"      # skip the end-to-end integration run
pytest --cov=src          # with coverage
```

## Python API

```python
from src.pipeline import Pipeline
from src.config import Config

pipeline = Pipeline(Config.load())
artifacts = pipeline.run("data/sample/sample_store_sales.csv", horizon=30)

print(artifacts.comparison)
print(artifacts.insights)
```
