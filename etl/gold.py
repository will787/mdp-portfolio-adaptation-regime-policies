from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import json 


ROOT = Path.cwd()
print(ROOT)

sys.path.insert(0, str(ROOT))
from config.paths import BRONZE_DIR, SILVER_DIR, GOLD_DIR, CONFIG_DIR

with open(CONFIG_DIR / "feature_config.json") as f:
    FEATURE_CONFIG = json.load(f)


daily = pd.read_parquet(SILVER_DIR / "daily.parquet")
monthly = pd.read_parquet(SILVER_DIR / "monthly.parquet")

def build_features(df, window_vol=21, window_mom=63, lags=[1, 3]):

    features = pd.DataFrame(index=df.index)

    for col in df.columns:

        cfg = FEATURE_CONFIG[col]

        if "transformations" not in cfg:
            print(col)
            print(cfg)
            raise ValueError(f"{col} sem transformations")
    
    for col in df.columns:

        cfg = FEATURE_CONFIG[col]
        s = df[col]

        returns = s.pct_change(fill_method=None)
        vol = returns.rolling(window_vol).std()
        rolling_mean = s.rolling(window_vol).mean()
        rolling_std = s.rolling(window_vol).std()

        for transform in cfg["transformations"]:

            if transform == "pct_change":
                for lag in lags:
                    features[f"{col}_pct_change_lag_{lag}m"] = s.pct_change(periods=lag, fill_method=None)

            elif transform == "returns":
                features[f"{col}_returns"] = returns

            elif transform == "z_score":
                features[f"{col}_zscore"] = (
                    s - rolling_mean
                ) / rolling_std

            elif transform == "volatily":
                features[f"{col}_volatily"] = vol

            elif transform == "vol_regime":
                features[f"{col}_vol_regime"] = (
                    vol / vol.rolling(window_vol).mean())

            elif transform == "momentum":
                features[f"{col}_momentum"] = (
                    s - s.rolling(window_mom).mean()) - 1

    return features


daily_features = build_features(daily)
daily_features_clean = daily_features.shift(1).fillna(0.0)

monthly_features = build_features(monthly)
monthly_features.index = pd.to_datetime(monthly_features.index)

monthly_aligned = monthly_features.reindex(daily_features.index).ffill().fillna(0.0)
monthly_aligned_clean = monthly_aligned.shift(1).fillna(0.0)


daily_features_clean = daily_features_clean.loc[:, ~daily_features_clean.columns.duplicated()]
monthly_aligned_clean = monthly_aligned_clean.loc[:, ~monthly_aligned_clean.columns.duplicated()]

gold_features = pd.concat([daily_features_clean, monthly_aligned_clean], axis=1)

gold_features.to_parquet(GOLD_DIR / "macro_features_hmm.parquet")
