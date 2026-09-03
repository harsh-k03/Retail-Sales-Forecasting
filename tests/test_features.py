"""Feature engineering, with explicit leakage guards."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import schema as S
from src.features.builder import FeatureBuilder
from src.features.calendar import FOURIER_EPOCH, add_calendar_features, future_frame
from src.features.encoders import UNKNOWN, OrdinalEncoder
from src.features.lags import add_lag_features, add_rolling_features


def test_calendar_features_match_the_date():
    frame = pd.DataFrame({S.DATE: pd.to_datetime(["2022-03-05", "2022-12-31"])})
    out = add_calendar_features(frame)
    assert out["dayofweek"].tolist() == [5, 5]
    assert out["is_weekend"].tolist() == [1, 1]
    assert out["is_month_end"].tolist() == [0, 1]


def test_lag_feature_equals_previous_value(clean_frame):
    out = add_lag_features(clean_frame, [1])
    group = out[(out[S.STORE] == out[S.STORE].iloc[0]) & (out[S.PRODUCT] == out[S.PRODUCT].iloc[0])]
    group = group.sort_values(S.DATE)
    assert np.allclose(group[f"{S.TARGET}_lag_1"].to_numpy()[1:], group[S.TARGET].to_numpy()[:-1])


def test_rolling_features_never_include_the_current_row():
    """A constant series with one spike must not see the spike on the spike's own row."""
    frame = pd.DataFrame(
        {
            S.DATE: pd.date_range("2022-01-01", periods=60, freq="D"),
            S.STORE: "S",
            S.PRODUCT: "P",
            S.TARGET: 10.0,
        }
    )
    frame.loc[30, S.TARGET] = 1000.0
    out = add_rolling_features(frame, [7], ["mean"]).sort_values(S.DATE).reset_index(drop=True)
    assert out.loc[30, f"{S.TARGET}_roll7_mean"] == 10.0
    assert out.loc[31, f"{S.TARGET}_roll7_mean"] > 10.0
    assert out.loc[38, f"{S.TARGET}_roll7_mean"] == 10.0  # spike has left the window


def test_builder_feature_names_all_exist(feature_frame, builder):
    assert len(builder.spec.feature_names) > 20
    assert set(builder.spec.feature_names) <= set(feature_frame.columns)


def test_no_lag_nans_after_warmup_drop(feature_frame, builder):
    lag_columns = [c for c in builder.spec.feature_names if c.startswith(f"{S.TARGET}_lag_")]
    assert feature_frame[lag_columns].isna().sum().sum() == 0


def test_target_is_not_a_feature(builder):
    assert S.TARGET not in builder.spec.feature_names


def test_fourier_phase_is_window_independent(clean_frame, config):
    builder = FeatureBuilder(config)
    full = builder.fit_transform(clean_frame)
    tail = builder.transform(clean_frame[clean_frame[S.DATE] >= clean_frame[S.DATE].max() - pd.Timedelta(days=60)])
    last_full = full[full[S.DATE] == full[S.DATE].max()]["sin_7_1"].iloc[0]
    last_tail = tail[tail[S.DATE] == tail[S.DATE].max()]["sin_7_1"].iloc[0]
    assert np.isclose(last_full, last_tail)
    assert FOURIER_EPOCH.year == 2000


def test_encoder_maps_unseen_categories_to_unknown():
    encoder = OrdinalEncoder(columns=["store"]).fit(pd.DataFrame({"store": ["a", "b"]}))
    out = encoder.transform(pd.DataFrame({"store": ["a", "zzz"]}))
    assert out["store_code"].tolist() == [0, UNKNOWN]


def test_future_frame_extends_every_series(clean_frame):
    future = future_frame(clean_frame, horizon=5)
    n_series = clean_frame.groupby([S.STORE, S.PRODUCT], observed=True).ngroups
    assert len(future) == n_series * 5
    assert future[S.TARGET].isna().all()
    assert future[S.DATE].min() > clean_frame[S.DATE].max()
