from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import os
import sys

ROOT = Path.cwd()
print(ROOT)

sys.path.insert(0, str(ROOT))
from config.paths import  CONFIG_DIR, FEATURES_DIR

def try_read_dir():
    try:
        BASE_DIR = Path(__file__).resolve().parents[2]
    except NameError:
        diretorio_atual =  Path.cwd()

        while_raiz = diretorio_atual
        while not (while_raiz / 'src').exists() and while_raiz.parent != while_raiz:
            while_raiz = while_raiz.parent
        
        BASE_DIR = while_raiz

    return BASE_DIR

def read_gold_data(gold_path: str, base_dir: Path = try_read_dir()):
    p = base_dir / gold_path
    if not p.exists():
        raise FileNotFoundError(f"Arquivo Gold não encontrado em: {gold_path}")
    return pd.read_parquet(p)

def build_feature_store(gold_df, features, start_date=None, end_date=None, name_feature_store='feature_store'):
    df = pd.read_parquet(gold_df)
    df = df.copy()
    if start_date:
        df = df.loc[start_date:]

    if end_date:
        df = df.loc[:end_date]

    df = df[features]

    df = df.sort_index()

    df = df.ffill()
    
    df = df.dropna()
    
    df.to_csv(FEATURES_DIR / f'{name_feature_store}.csv', index=True)
    return df
