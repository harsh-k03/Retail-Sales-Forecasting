"""Turn model output and history into ranked, actionable business recommendations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data import schema as S
from src.eda import analysis
from src.logger import get_logger

logger = get_logger(__name__)

HIGH, MEDIUM, LOW = "high", "medium", "low"


@dataclass
class Insight:
    category: str
    title: str
    detail: str
    action: str
    priority: str = MEDIUM
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InsightEngine:
    """Rule-based engine over history + forecast. Every rule cites its evidence."""

    def __init__(
        self,
        history: pd.DataFrame,
        forecast: Optional[pd.DataFrame] = None,
        importance: Optional[pd.DataFrame] = None,
    ) -> None:
        self.history = history
        self.forecast = forecast
        self.importance = importance

    def run(self) -> List[Insight]:
        insights: List[Insight] = []
        for rule in (
            self._demand_direction,
            self._inventory_pressure,
            self._promotion_efficiency,
            self._store_concentration,
            self._weekday_staffing,
            self._seasonal_peaks,
            self._volatile_series,
            self._underperforming_products,
            self._anomaly_watch,
            self._driver_summary,
        ):
            try:
                insights.extend(rule())
            except Exception as exc:
                logger.warning("Insight rule %s failed: %s", rule.__name__, exc)
        order = {HIGH: 0, MEDIUM: 1, LOW: 2}
        return sorted(insights, key=lambda i: order.get(i.priority, 3))

    def to_frame(self) -> pd.DataFrame:
        rows = [i.to_dict() for i in self.run()]
        if not rows:
            return pd.DataFrame(columns=["category", "title", "detail", "action", "priority"])
        return pd.DataFrame(rows)[["priority", "category", "title", "detail", "action"]]

    # ------------------------------------------------------------------ rules
    def _demand_direction(self) -> List[Insight]:
        if self.forecast is None or self.forecast.empty:
            return []
        horizon = int(self.forecast["step"].max())
        forecast_total = float(self.forecast["forecast"].sum())
        recent = self.history[self.history[S.DATE] > self.history[S.DATE].max() - pd.Timedelta(days=horizon)]
        recent_total = float(recent[S.TARGET].sum())
        if not recent_total:
            return []
        change = (forecast_total / recent_total - 1) * 100
        if abs(change) < 3:
            return [Insight("demand", "Demand is flat",
                            f"Next {horizon} days track within {change:+.1f}% of the last {horizon} days.",
                            "Hold current inventory and staffing plans.", LOW,
                            {"change_pct": round(change, 2)})]
        rising = change > 0
        return [Insight(
            "demand",
            f"Demand is forecast to {'rise' if rising else 'fall'} {abs(change):.1f}%",
            f"Next {horizon} days total {forecast_total:,.0f} vs {recent_total:,.0f} in the last {horizon} days.",
            "Raise purchase orders and shift staffing to peak days." if rising
            else "Trim replenishment and plan clearance on slow-moving lines.",
            HIGH if abs(change) > 10 else MEDIUM,
            {"change_pct": round(change, 2), "forecast_total": round(forecast_total, 2)},
        )]

    def _inventory_pressure(self) -> List[Insight]:
        if self.forecast is None or self.forecast.empty:
            return []
        by_store = self.forecast.groupby(S.STORE, observed=True)["forecast"].sum()
        horizon = int(self.forecast["step"].max())
        recent = self.history[self.history[S.DATE] > self.history[S.DATE].max() - pd.Timedelta(days=horizon)]
        base = recent.groupby(S.STORE, observed=True)[S.TARGET].sum()
        delta = ((by_store / base.reindex(by_store.index) - 1) * 100).dropna().sort_values()
        out: List[Insight] = []
        if len(delta) and delta.iloc[-1] > 8:
            store = delta.index[-1]
            out.append(Insight("inventory", f"Store {store} needs more stock",
                               f"Forecast is {delta.iloc[-1]:+.1f}% above its recent run-rate.",
                               f"Increase replenishment for store {store} ahead of the window.", HIGH,
                               {"store": store, "change_pct": round(float(delta.iloc[-1]), 2)}))
        if len(delta) and delta.iloc[0] < -8:
            store = delta.index[0]
            out.append(Insight("inventory", f"Store {store} risks overstock",
                               f"Forecast is {delta.iloc[0]:+.1f}% below its recent run-rate.",
                               f"Delay replenishment for store {store} and review markdowns.", MEDIUM,
                               {"store": store, "change_pct": round(float(delta.iloc[0]), 2)}))
        return out

    def _promotion_efficiency(self) -> List[Insight]:
        table = analysis.promotion_effect(self.history)
        if table is None or "uplift_pct" not in table.columns or len(table) < 2:
            return []
        uplift = float(table["uplift_pct"].iloc[-1])
        promo_days = int(table.loc[table["on_promotion"] == 1, "count"].iloc[0])
        share = promo_days / max(len(self.history), 1) * 100
        if uplift < 5:
            return [Insight("promotion", "Promotions are barely lifting sales",
                            f"Promotion days average only {uplift:+.1f}% above baseline across {promo_days:,} rows.",
                            "Rework promo depth/targeting before the next cycle; the current mix is diluting margin.",
                            HIGH, {"uplift_pct": round(uplift, 2), "promo_share_pct": round(share, 2)})]
        return [Insight("promotion", f"Promotions lift sales {uplift:.1f}%",
                        f"Measured across {promo_days:,} promoted rows ({share:.1f}% of days).",
                        "Concentrate promo budget on the highest-uplift product families and peak weekdays.",
                        MEDIUM, {"uplift_pct": round(uplift, 2), "promo_share_pct": round(share, 2)})]

    def _store_concentration(self) -> List[Insight]:
        stores = analysis.store_performance(self.history)
        if len(stores) < 2:
            return []
        top_share = float(stores["share_pct"].iloc[0])
        if top_share < 100 / len(stores) * 1.6:
            return []
        return [Insight("network", f"Store {stores[S.STORE].iloc[0]} drives {top_share:.1f}% of revenue",
                        f"Top store is {top_share / (100 / len(stores)):.1f}x the network average.",
                        "Protect service levels at this store first; replicate its assortment at weaker sites.",
                        MEDIUM, {"store": stores[S.STORE].iloc[0], "share_pct": round(top_share, 2)})]

    def _weekday_staffing(self) -> List[Insight]:
        dow = analysis.seasonality_table(self.history)["dayofweek"].dropna()
        if dow.empty:
            return []
        best = dow.loc[dow[S.TARGET].idxmax()]
        worst = dow.loc[dow[S.TARGET].idxmin()]
        gap = (best[S.TARGET] / worst[S.TARGET] - 1) * 100 if worst[S.TARGET] else 0
        if gap < 10:
            return []
        return [Insight("staffing", f"{best['dayofweek']} runs {gap:.0f}% above {worst['dayofweek']}",
                        f"Average sales {best[S.TARGET]:,.0f} vs {worst[S.TARGET]:,.0f}.",
                        f"Move shift hours from {worst['dayofweek']} to {best['dayofweek']}.",
                        MEDIUM, {"gap_pct": round(float(gap), 2)})]

    def _seasonal_peaks(self) -> List[Insight]:
        month = analysis.seasonality_table(self.history)["month"]
        if len(month) < 6:
            return []
        peak = month.loc[month[S.TARGET].idxmax()]
        avg = month[S.TARGET].mean()
        lift = (peak[S.TARGET] / avg - 1) * 100 if avg else 0
        if lift < 8:
            return []
        name = pd.Timestamp(2000, int(peak["month"]), 1).strftime("%B")
        return [Insight("seasonality", f"{name} is the seasonal peak",
                        f"Averages {lift:+.1f}% above the typical month.",
                        f"Lock in supplier capacity and seasonal hiring ~6 weeks before {name}.",
                        MEDIUM, {"peak_month": name, "lift_pct": round(float(lift), 2)})]

    def _volatile_series(self) -> List[Insight]:
        stability = analysis.series_stability(self.history)
        risky = stability[stability["cv"] > 0.6]
        if risky.empty:
            return []
        top = risky.iloc[0]
        return [Insight("risk", f"{len(risky)} store-product series are highly volatile",
                        f"Worst is {top[S.STORE]} / {top[S.PRODUCT]} with CV {top['cv']:.2f}.",
                        "Hold extra safety stock on these lines; forecasts there carry wider error bands.",
                        MEDIUM, {"n_volatile": int(len(risky))})]

    def _underperforming_products(self) -> List[Insight]:
        products = analysis.product_performance(self.history, top_n=1000)
        if len(products) < 3:
            return []
        tail = products.tail(max(1, len(products) // 4))
        share = float(tail["share_pct"].sum())
        if share > 12:
            return []
        return [Insight("assortment", f"Bottom {len(tail)} products contribute {share:.1f}% of sales",
                        "The long tail ties up shelf space and working capital.",
                        "Review the tail for delisting or reduced facings at low-volume stores.",
                        LOW, {"tail_share_pct": round(share, 2)})]

    def _anomaly_watch(self) -> List[Insight]:
        anomalies = analysis.detect_anomalies(self.history)
        if anomalies.empty:
            return []
        recent = anomalies[anomalies[S.DATE] > self.history[S.DATE].max() - pd.Timedelta(days=90)]
        if recent.empty:
            return []
        worst = recent.iloc[0]
        return [Insight("data_quality", f"{len(recent)} anomalous days in the last 90 days",
                        f"Largest is {worst[S.DATE].date()} ({worst['direction']}, z={worst['z_score']:.1f}).",
                        "Confirm these are real events (closures, stockouts) before trusting the forecast there.",
                        HIGH if len(recent) > 5 else LOW, {"n_anomalies": int(len(recent))})]

    def _driver_summary(self) -> List[Insight]:
        if self.importance is None or self.importance.empty:
            return []
        col = "mean_abs_shap" if "mean_abs_shap" in self.importance.columns else "importance"
        top = self.importance.head(3)["feature"].tolist()
        return [Insight("model", "Top forecast drivers",
                        f"The model leans hardest on {', '.join(top)}.",
                        "Keep these inputs clean and available at forecast time; they carry most of the signal.",
                        LOW, {"top_features": top, "metric": col})]


def generate_insights(history, forecast=None, importance=None) -> pd.DataFrame:
    return InsightEngine(history, forecast, importance).to_frame()
