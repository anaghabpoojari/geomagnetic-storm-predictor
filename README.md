# 🌌 Geomagnetic Storm Intensity Predictor  
Predict the intensity of geomagnetic storms using solar activity data with a Random Forest ML model, exposed via a Flask API, and a simple frontend for real-time predictions.  
---  
Geomagnetic storms are disturbances in the Earth's magnetosphere caused by solar activity such as solar flares or coronal mass ejections (CME). Accurately predicting the intensity of these storms is critical for satellite operations, power grids, and communication systems.  
## This project:  
- Trains a **Random Forest Regressor** on historical geomagnetic storm data.  
- Uses feature engineering to handle time-series and categorical variables.  
- Exposes a **Flask API** for real-time predictions.  
- Provides a **frontend web interface** where users can input solar activity parameters and get predicted KP Index values instantly.  
---  
## Tech-Stack 
- Python 3.x  
- scikit-learn (Random Forest Regressor)  
- pandas & numpy  
- Flask + Flask-CORS  
- HTML, CSS, JavaScript 
## 📂 Project Structure  
geomagnetic-storm-predictor/
│
├─ backend/ # Flask backend
│ ├─ app.py # Flask API code
│ ├─ geomagnetic_rf_model_v2.pkl
│ ├─ geomagnetic_scaler_v2.pkl
│ └─ geomagnetic_ohe_v2.pkl
│
├─ frontend/ # Web frontend
│ ├─ index.html
│ └─ script.js
│
├─ train_model.py # ML training script
├─ dataset/ # Input CSV dataset
├─ .gitignore
└─ README.md
