import streamlit as st
import pandas as pd
import fastf1
from main import load_race_data, train_model

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️")

st.title("🏎️ F1 Race Predictor")
st.markdown("Predict race results based on qualifying times and past performance.")

# --- User Input ---
year = st.number_input("Enter the race year:", min_value=2018, max_value=2025, value=2024)
race_name = st.text_input("Enter the race name:", value="China").title()

# --- Load session and get drivers ---
driver_codes = []
if "last_year" not in st.session_state or st.session_state.last_year != year or st.session_state.last_race != race_name:
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load()
        if session.laps.empty:
            raise ValueError("No lap data available for the selected session.")
        driver_codes = sorted(session.laps['Driver'].dropna().unique())
        st.session_state["drivers"] = driver_codes
        st.session_state.last_year = year
        st.session_state.last_race = race_name
        st.success(f"✅ Loaded drivers for {race_name} {year}.")
    except Exception as e:
        st.error(f"❌ Could not load drivers: {e}")

# --- Qualifying Times Input ---
if "drivers" in st.session_state:
    st.subheader("🚅️ Enter Qualifying Times (in seconds)")
    quali_data = []
    for code in st.session_state["drivers"]:
        time = st.text_input(f"{code}", key=code)
        if time.strip():
            try:
                quali_data.append({"Driver": code, "QualifyingTime (s)": float(time)})
            except ValueError:
                st.warning(f"⚠️ Invalid input for {code}")

    # --- Predict Button ---
    if st.button("🔮 Predict Race Results") and quali_data:
        try:
            with st.spinner("Loading race data..."):
                laps, sector_times = load_race_data(year, race_name)

            quali_df = pd.DataFrame(quali_data)
            results, mae = train_model(laps, quali_df, sector_times)

            # Sort results by predicted race time
            results = results.sort_values(by="PredictedRaceTime (s)")

            st.success(f"✅ Prediction complete for {race_name} GP {year+1}!")
            st.subheader("🌟 Predicted Race Results")
            st.dataframe(results[["Driver", "QualifyingTime (s)", "PredictedRaceTime (s)"]])

            st.markdown(f"📊 **Model MAE (Mean Absolute Error)**: `{mae:.2f}` seconds")

        except Exception as e:
            st.error(f"❌ Error loading race data or predicting: {e}")
