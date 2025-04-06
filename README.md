🏁 F1 Race Predictor


This project predicts the finishing times of Formula 1 drivers based on qualifying times and historical race sector performance using machine learning.
It uses the FastF1 API to gather race data and applies a Gradient Boosting Regressor model for predictions.


------------------------------------------------------------------------------------------

PROJECT STRUCTURE :

```
f1-race-predictor/
├── app.py                  # Streamlit UI 
├── main.py                 # Core ML logic and data loading
├── f1_cache/               # FastF1 cache directory (created automatically)
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```



------------------------------------------------------------------------------------------

🔧 Clone the Repository


git clone https://github.com/yourusername/f1-race-predictor.git


cd f1-race-predictor

------------------------------------------------------------------------------------------

📦 Create Virtual Environment (Recommended)

python -m venv venv


source venv/bin/activate  # On Windows: venv\Scripts\activate


------------------------------------------------------------------------------------------

🛠️ Install Dependencies


pip install -r requirements.txt




------------------------------------------------------------------------------------------

📦 Dependencies


You can also install packages manually if needed:


pip install streamlit fastf1 scikit-learn pandas numpy


------------------------------------------------------------------------------------------

🚀 How to Run <br/>

🌐 Launch the Streamlit App </br>


streamlit run app.py



------------------------------------------------------------------------------------------

🧠 How It Works:


User Input: You select the race year and race name.


Driver Fetch: FastF1 pulls all driver codes from the selected race.


Qualifying Input: You enter each driver’s qualifying time (in seconds).


Data Merge: The app fetches historical sector times and merges them with qualifying input.


Model Training: A Gradient Boosting Regressor is trained using lap and sector time data.


Prediction: Race times are predicted, sorted, and displayed.



------------------------------------------------------------------------------------------

❓ Example Use


Choose 2024 as year


Enter race name: China


Wait for drivers to load automatically


Enter each driver’s qualifying time


Click "🔮 Predict Race Results" to view predictions.



------------------------------------------------------------------------------------------

👨‍💻 Developer Info


Language: Python 3.8+


Libraries: FastF1, Streamlit, scikit-learn, pandas, numpy


Model: GradientBoostingRegressor












