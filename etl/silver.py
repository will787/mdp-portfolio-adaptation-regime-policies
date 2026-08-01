
from pathlib import Path
import os
import sys
import pandas as pd
import json 


ROOT = Path.cwd()
print(ROOT)

sys.path.insert(0, str(ROOT))
from config.paths import BRONZE_DIR, SILVER_DIR, CONFIG_DIR


with open(CONFIG_DIR / "feature_config.json") as f:
    FEATURE_CONFIG = json.load(f)

def filter_bussines_daily(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[df.index.dayofweek < 5]

    if 'ibovespa_br' in df.columns:
        df = df[df["ibovespa_br"].notna()]

    return df


def split_by_frequency(df, feature_cfg):

    daily = []
    monthly = []

    for col in df.columns:

        if col == "data":
            continue

        cfg = feature_cfg.get(col)

        if cfg is None:
            print(f"{col} não está no feature_config")
            continue

        if cfg["frequency"] == "daily":
            daily.append(col)

        elif cfg["frequency"] == "monthly":
            monthly.append(col)

    daily_df = df[["data"] + daily]
    monthly_df = df[["data"] + monthly]

    return daily_df, monthly_df


bronze_files = list(BRONZE_DIR.glob("*.csv"))
daily_views = []
monthly_views = []

for base in bronze_files:

    df = pd.read_csv(base)

    daily_df, monthly_df = split_by_frequency(df, FEATURE_CONFIG)

    if len(daily_df.columns) > 1:
        daily_views.append(daily_df)

    if len(monthly_df.columns) > 1:
        monthly_views.append(monthly_df)

daily = (
    pd.concat(
        [df.set_index("data") for df in daily_views],
        axis=1
    )
    .sort_index()
)

daily = filter_bussines_daily(daily)


monthly = (
    pd.concat(
        [df.set_index("data") for df in monthly_views],
        axis=1
    )
    .sort_index()
)

for col in monthly.columns:
    cfg = FEATURE_CONFIG.get(col, {})
    lag = cfg.get("lag", 0)

    if lag > 0:
        monthly[col] = monthly[col].shift(lag)
        print(f" -> Lag de {lag} mês(es) aplicado na coluna: {col}")

daily.to_parquet(SILVER_DIR / "daily.parquet")
monthly.to_parquet(SILVER_DIR / "monthly.parquet")
