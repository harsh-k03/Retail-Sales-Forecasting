"""SHAP-based global and local explanations for tree forecasters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.logger import get_logger
from src.models.base import BaseForecaster

logger = get_logger(__name__)


@dataclass
class Explanation:
    values: np.ndarray
    base_value: float
    sample: pd.DataFrame
    feature_names: List[str]

    def global_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        mean_abs = np.abs(self.values).mean(axis=0)
        out = (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        out["share_pct"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum() * 100
        return out.head(top_n) if top_n else out

    def local(self, row: int = 0, top_n: int = 12) -> pd.DataFrame:
        contrib = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": self.sample.iloc[row].to_numpy(),
                "shap_value": self.values[row],
            }
        )
        contrib["abs"] = contrib["shap_value"].abs()
        return contrib.sort_values("abs", ascending=False).head(top_n).drop(columns="abs").reset_index(drop=True)

    def direction_table(self, top_n: int = 15) -> pd.DataFrame:
        """Signed average effect - answers 'does this driver push sales up or down'."""
        mean_signed = self.values.mean(axis=0)
        mean_abs = np.abs(self.values).mean(axis=0)
        out = pd.DataFrame(
            {"feature": self.feature_names, "mean_shap": mean_signed, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        out["direction"] = np.where(out["mean_shap"] >= 0, "increases sales", "decreases sales")
        return out.head(top_n).reset_index(drop=True)


def explain(
    model: BaseForecaster,
    X: pd.DataFrame,
    max_samples: int = 2000,
    seed: int = 42,
) -> Optional[Explanation]:
    """Compute SHAP values on a sample of rows. Returns None for unsupported models."""
    if not getattr(model, "supports_shap", False) or model.model is None:
        logger.info("SHAP not supported for %s", model.name)
        return None
    import shap

    sample = X if len(X) <= max_samples else X.sample(max_samples, random_state=seed)
    sample = sample[model.feature_names_].astype(float)
    try:
        explainer = shap.TreeExplainer(model.model)
        values = explainer.shap_values(sample)
    except Exception as exc:
        logger.warning("TreeExplainer failed (%s); falling back to permutation sampling", exc)
        return None

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 0]
    base = explainer.expected_value
    base_value = float(np.ravel(base)[0]) if base is not None else 0.0
    return Explanation(values=values, base_value=base_value, sample=sample, feature_names=list(sample.columns))


def save_summary_plot(exp: Explanation, out_dir: Path, top_n: int = 20) -> Path:
    import shap

    path = Path(out_dir) / "shap_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(exp.values, exp.sample, max_display=top_n, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def save_bar_plot(exp: Explanation, out_dir: Path, top_n: int = 20) -> Path:
    import seaborn as sns

    path = Path(out_dir) / "shap_importance.png"
    data = exp.global_importance(top_n)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(data))))
    sns.barplot(data=data, y="feature", x="mean_abs_shap", ax=ax)
    ax.set_title("Global feature importance (mean |SHAP|)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_dependence_plot(exp: Explanation, feature: str, out_dir: Path) -> Optional[Path]:
    import shap

    if feature not in exp.feature_names:
        return None
    path = Path(out_dir) / f"shap_dependence_{feature}.png"
    plt.figure()
    shap.dependence_plot(feature, exp.values, exp.sample, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def save_waterfall_plot(exp: Explanation, row: int, out_dir: Path, top_n: int = 12) -> Path:
    import shap

    path = Path(out_dir) / "shap_waterfall.png"
    explanation = shap.Explanation(
        values=exp.values[row],
        base_values=exp.base_value,
        data=exp.sample.iloc[row].to_numpy(),
        feature_names=exp.feature_names,
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=top_n, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def generate_all(exp: Explanation, out_dir: Path, top_features: int = 20) -> Dict[str, str]:
    out_dir = Path(out_dir)
    figures = {"shap_importance": str(save_bar_plot(exp, out_dir, top_features))}
    for name, fn in (("shap_summary", save_summary_plot), ("shap_waterfall", lambda e, d: save_waterfall_plot(e, 0, d))):
        try:
            figures[name] = str(fn(exp, out_dir))
        except Exception as exc:
            logger.warning("Could not render %s: %s", name, exc)
    top = exp.global_importance(3)["feature"].tolist()
    for feature in top:
        try:
            path = save_dependence_plot(exp, feature, out_dir)
            if path:
                figures[f"shap_dependence_{feature}"] = str(path)
        except Exception as exc:
            logger.warning("Dependence plot failed for %s: %s", feature, exc)
    return figures
