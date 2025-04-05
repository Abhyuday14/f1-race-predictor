🏎️ F1 Race Predictor App
A Streamlit-based web application that predicts Formula 1 race outcomes based on qualifying times and historical performance using real-time data from the FastF1 API.

🔍 Overview
This app allows you to:

Select a race by year and Grand Prix name.

Load actual driver data from the FastF1 API for that race.

Manually input qualifying times for each driver.

Use a machine learning model (Gradient Boosting Regressor) to predict race performance.

View the predicted race order along with estimated race time.

🚀 How It Works
You enter the race year and Grand Prix name (e.g., China, Bahrain).

The app loads lap data from that race using FastF1 and displays all drivers.

You enter each driver’s qualifying time (in seconds).

The app trains a model based on historical lap data and uses it to predict each driver's race time.

The results are displayed in the predicted order of finish, along with the model's accuracy (MAE).

🛠️ Tech Stack
Frontend: Streamlit

Backend: FastF1, scikit-learn

Language: Python 3
