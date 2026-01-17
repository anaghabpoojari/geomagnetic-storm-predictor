from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

# Load trained artifacts
model = joblib.load("geomagnetic_rf_model_v2.pkl")
scaler = joblib.load("geomagnetic_scaler_v2.pkl")
ohe = joblib.load("geomagnetic_ohe_v2.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Geomagnetic Storm Prediction API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Numeric inputs
        duration_hours = float(data["duration_hours"])
        hour_of_day = int(data["hour_of_day"])
        day_of_year = int(data["day_of_year"])
        kp_index_lag1 = float(data.get("kp_index_lag1", 0))

        # Categorical inputs
        event_type = data["event_type"]
        class_type = data["class_type"]

        # One-hot encode categorical features
        cat_df = pd.DataFrame(
            [[event_type, class_type]],
            columns=["event_type", "class_type"]
        )
        cat_encoded = ohe.transform(cat_df)

        # Combine features
        features = np.hstack([
            [duration_hours, hour_of_day, day_of_year, kp_index_lag1],
            cat_encoded[0]
        ])

        # Scale
        features_scaled = scaler.transform([features])

        # Predict
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            "predicted_kp_index": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
