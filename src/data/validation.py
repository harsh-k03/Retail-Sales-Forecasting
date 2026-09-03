"""Schema and quality validation with a machine-readable report."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import Config
from src.data import schema as S
from src.logger import get_logger
from src.utils import save_json

logger = get_logger(__name__)

ERROR = "error"
WARNING = "warning"


@dataclass
class Issue:
    rule: str
    severity: str
    message: str
    count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    passed: bool
    n_rows: int
    n_columns: int
    issues: List[Issue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["n_errors"] = len(self.errors)
        payload["n_warnings"] = len(self.warnings)
        return payload

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["rule", "severity", "message", "count"])
        return pd.DataFrame([asdict(i) for i in self.issues])[
            ["rule", "severity", "message", "count"]
        ]

    def save(self, path: Path) -> Path:
        return save_json(self.to_dict(), Path(path))


def validate(frame: pd.DataFrame, config: Optional[Config] = None) -> ValidationReport:
    """Run every validation rule and return a report."""
    config = config or Config.load()
    rules = config.get("validation", {}) or {}
    issues: List[Issue] = []

    missing = [c for c in S.REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        issues.append(Issue("required_columns", ERROR, f"Missing required columns: {missing}", len(missing)))
        return ValidationReport(False, len(frame), frame.shape[1], issues)

    min_rows = int(rules.get("min_rows", 0))
    if len(frame) < min_rows:
        issues.append(Issue("min_rows", ERROR, f"Only {len(frame)} rows, need >= {min_rows}", len(frame)))

    # dates
    bad_dates = int(frame[S.DATE].isna().sum())
    if bad_dates:
        issues.append(Issue("invalid_dates", ERROR, f"{bad_dates} unparseable dates", bad_dates))
    min_date = pd.Timestamp(rules.get("min_date", "1900-01-01"))
    too_old = int((frame[S.DATE] < min_date).sum())
    if too_old:
        issues.append(Issue("date_range", WARNING, f"{too_old} rows before {min_date.date()}", too_old))
    cutoff = pd.Timestamp.today().normalize() + pd.Timedelta(days=int(rules.get("max_future_days", 0)))
    future = int((frame[S.DATE] > cutoff).sum())
    if future:
        issues.append(Issue("future_dates", WARNING, f"{future} rows dated after {cutoff.date()}", future))

    # target
    missing_target = int(frame[S.TARGET].isna().sum())
    ratio = missing_target / max(len(frame), 1)
    if ratio > float(rules.get("max_missing_target_ratio", 0.05)):
        issues.append(Issue("missing_target", ERROR, f"{missing_target} missing target values ({ratio:.1%})", missing_target))
    elif missing_target:
        issues.append(Issue("missing_target", WARNING, f"{missing_target} missing target values", missing_target))

    negatives = int((frame[S.TARGET] < 0).sum())
    if negatives and not rules.get("allow_negative_sales", False):
        issues.append(Issue("negative_sales", WARNING, f"{negatives} negative sales values", negatives))

    if pd.api.types.is_numeric_dtype(frame[S.TARGET]) is False:
        issues.append(Issue("target_dtype", ERROR, "Target column is not numeric", 0))

    # duplicates on the natural key
    dupes = int(frame.duplicated(subset=S.KEY_COLUMNS).sum())
    dupe_ratio = dupes / max(len(frame), 1)
    if dupe_ratio > float(rules.get("max_duplicate_ratio", 0.01)):
        issues.append(Issue("duplicates", ERROR, f"{dupes} duplicate (date, store, product) rows", dupes))
    elif dupes:
        issues.append(Issue("duplicates", WARNING, f"{dupes} duplicate (date, store, product) rows", dupes))

    # identifier hygiene
    for col in (S.STORE, S.PRODUCT):
        blanks = int(frame[col].isna().sum() + (frame[col].astype("string").str.strip() == "").sum())
        if blanks:
            issues.append(Issue(f"blank_{col}", ERROR, f"{blanks} blank {col} identifiers", blanks))

    # calendar continuity per series
    gaps = _gap_summary(frame)
    if gaps["series_with_gaps"]:
        issues.append(
            Issue(
                "date_gaps",
                WARNING,
                f"{gaps['series_with_gaps']} series have missing dates",
                gaps["series_with_gaps"],
                {"max_gap_days": gaps["max_gap_days"]},
            )
        )

    report = ValidationReport(
        passed=not any(i.severity == ERROR for i in issues),
        n_rows=len(frame),
        n_columns=frame.shape[1],
        issues=issues,
        summary=summarize(frame),
    )
    logger.info("Validation %s: %d errors, %d warnings", "passed" if report.passed else "FAILED",
                len(report.errors), len(report.warnings))
    return report


def _gap_summary(frame: pd.DataFrame) -> Dict[str, int]:
    grouped = frame.dropna(subset=[S.DATE]).groupby([S.STORE, S.PRODUCT])[S.DATE]
    with_gaps, max_gap = 0, 0
    for _, dates in grouped:
        if len(dates) < 3:
            continue
        diffs = dates.sort_values().diff().dt.days.dropna()
        if len(diffs) and diffs.max() > diffs.median():
            if diffs.max() > 1:
                with_gaps += 1
                max_gap = max(max_gap, int(diffs.max()))
    return {"series_with_gaps": with_gaps, "max_gap_days": max_gap}


def summarize(frame: pd.DataFrame) -> Dict[str, Any]:
    """Dataset-level facts used by the dashboard and reports."""
    out: Dict[str, Any] = {
        "rows": len(frame),
        "columns": list(frame.columns),
        "optional_present": S.present_optional(frame.columns),
        "missing_by_column": {c: int(frame[c].isna().sum()) for c in frame.columns},
    }
    if S.DATE in frame and frame[S.DATE].notna().any():
        out["date_min"] = str(frame[S.DATE].min().date())
        out["date_max"] = str(frame[S.DATE].max().date())
        out["n_days"] = int(frame[S.DATE].nunique())
    for col in (S.STORE, S.PRODUCT):
        if col in frame:
            out[f"n_{col}s"] = int(frame[col].nunique())
    if S.TARGET in frame:
        target = frame[S.TARGET].dropna()
        out["target"] = {
            "mean": float(target.mean()) if len(target) else 0.0,
            "std": float(target.std()) if len(target) else 0.0,
            "min": float(target.min()) if len(target) else 0.0,
            "max": float(target.max()) if len(target) else 0.0,
            "zero_ratio": float((target == 0).mean()) if len(target) else 0.0,
        }
    return out
