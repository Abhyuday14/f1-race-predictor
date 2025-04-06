import streamlit as st
import pandas as pd
import fastf1
from main import load_race_data, train_model, driver_mapping

# Enable FastF1 cache early
fastf1.Cache.enable_cache('f1_cache')

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️")

st.title("🏁 F1 Race Predictor")
st.markdown("Predict race results based on qualifying times and past performance.")

# --- User Input ---
year = st.number_input("Enter the race year:", min_value=2024, max_value=2025, value=2024)
race_name = st.text_input("Enter the race name:", value="China").title()

# Qualifying Times Input
st.subheader("📥 Enter Qualifying Times (in seconds)")
quali_data = []
for driver in driver_mapping:
    time = st.text_input(f"{driver}", key=driver)
    if time.strip():
        try:
            quali_data.append({"Driver": driver, "QualifyingTime (s)": float(time)})
        except ValueError:
            st.warning(f"⚠️ Invalid input for {driver}")

# Predict Button
if st.button("🔮 Predict Race Results") and quali_data:
    try:
        with st.spinner("Loading race data..."):
            laps, sector_times = load_race_data(year, race_name)

        quali_df = pd.DataFrame(quali_data)
        results, mae = train_model(laps, quali_df, sector_times)

        results = results.sort_values(by="PredictedRaceTime (s)")

        st.success(f"✅ Prediction complete for {race_name} GP {year+1}!")
        st.subheader("🌟 Predicted Race Results")
        st.dataframe(results[["Driver", "QualifyingTime (s)", "PredictedRaceTime (s)"]])
        st.markdown(f"📊 **Model MAE (Mean Absolute Error)**: `{mae:.2f}` seconds")

    except Exception as e:
        st.error(f"❌ Error loading race data or predicting: {e}")
