"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import forecast, health, metadata, upload
from src.config import Config
from src.logger import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
REST interface to the retail sales forecasting platform.

* `POST /data/upload` - upload a CSV and get a validation report
* `POST /forecast` - multi-horizon forecast, optionally with scenario overrides
* `POST /insights` - ranked business recommendations
* `GET /model/metadata` - trained model contract and CV metrics
* `GET /predictions/history` - recent requests served by this instance
"""


def create_app() -> FastAPI:
    config = Config.load()
    configure_logging(
        level=config.get("logging.level", "INFO"),
        log_file=config.get("logging.file", "logs/app.log"),
    )
    application = FastAPI(
        title="Retail Sales Forecasting API",
        description=DESCRIPTION,
        version=str(config.get("project.version", "1.0.0")),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in (health.router, upload.router, forecast.router, metadata.router):
        application.include_router(router)

    @application.get("/", tags=["health"])
    def root() -> dict:
        return {"service": "retail-sales-forecasting", "docs": "/docs", "health": "/health"}

    logger.info("API ready")
    return application


app = create_app()
