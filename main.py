import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

def load_race_data(year, race_name):
    session = fastf1.get_session(year, race_name, "R")
    session.load()

    # Extract and clean lap data
    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps.dropna(inplace=True)

    # Convert timedelta to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()

    # Calculate mean sector times per driver
    sector_times = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

    return laps, sector_times

def train_model(laps, qualifying_df, sector_times):
    # Merge qualifying and sector times
    merged = qualifying_df.merge(sector_times, on="Driver", how="left")

    # Features
    X = merged[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
    
    # Targets
    y = laps.groupby("Driver")["LapTime (s)"].mean().reset_index()
    y = qualifying_df.merge(y, on="Driver", how="left")["LapTime (s)"].fillna(X["QualifyingTime (s)"])  # fallback if not found

    # Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predict on full dataset
    predictions = model.predict(X)
    qualifying_df["PredictedRaceTime (s)"] = predictions

    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return qualifying_df, mae
