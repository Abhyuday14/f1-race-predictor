import fastf1
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

def load_race_data(year, race_name):
    session = fastf1.get_session(year, race_name, "R")
    session.load()
    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps.dropna(inplace=True)

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()

    sector_times = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
    return laps, sector_times

def train_model(laps, quali_df, sector_times):
    quali_df["DriverCode"] = quali_df["Driver"].map(driver_mapping)
    merged = quali_df.merge(sector_times, left_on="DriverCode", right_on="Driver", how="left")
    X = merged[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
    y = laps.groupby("Driver")["LapTime (s)"].mean().reset_index()["LapTime (s)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
    model.fit(X_train, y_train)
    predictions = model.predict(X)
    mae = mean_absolute_error(y_test, model.predict(X_test))

    quali_df["PredictedRaceTime (s)"] = predictions
    quali_df = quali_df.sort_values(by="PredictedRaceTime (s)")
    return quali_df, mae
