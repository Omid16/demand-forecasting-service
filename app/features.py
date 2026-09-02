import pandas as pd

def make_features(y: pd.Series) -> pd.DataFrame:
    df = y.to_frame(name="y")
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["is_low_demand"] = (df["hour"] >= 23) | (df["hour"] <= 5)
    df["lag_168"] = df["y"].shift(168)
    df["ma_24"] = df["y"].rolling(24).mean().shift(25)

    return df.drop(columns="y")

