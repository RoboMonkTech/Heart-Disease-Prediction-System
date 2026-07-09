from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ==========================================
# Load Trained Model
# ==========================================

MODEL_PATH = os.path.join("model", "heart_pipeline.pkl")
model = joblib.load(MODEL_PATH)

# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        patient = pd.DataFrame([{

            "age": float(request.form["age"]),
            "sex": int(request.form["sex"]),
            "cp": int(request.form["cp"]),
            "trestbps": float(request.form["trestbps"]),
            "chol": float(request.form["chol"]),
            "fbs": int(request.form["fbs"]),
            "restecg": int(request.form["restecg"]),
            "thalach": float(request.form["thalach"]),
            "exang": int(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "slope": int(request.form["slope"]),
            "ca": int(request.form["ca"]),
            "thal": int(request.form["thal"])

        }])

        prediction = model.predict(patient)[0]

        probability = model.predict_proba(patient)[0]

        confidence = round(max(probability) * 100, 2)

        if prediction == 1:

            result = "❤️ High Risk of Heart Disease"
            risk = "High"

            recommendation = [
                "Consult a Cardiologist",
                "Monitor Blood Pressure Regularly",
                "Maintain a Healthy Diet",
                "Exercise at Least 30 Minutes Daily",
                "Reduce Cholesterol and Sodium Intake"
            ]

        else:

            result = "✅ Low Risk of Heart Disease"
            risk = "Low"

            recommendation = [
                "Continue a Healthy Lifestyle",
                "Exercise Regularly",
                "Maintain a Balanced Diet",
                "Schedule Annual Health Check-ups",
                "Avoid Smoking and Excess Alcohol"
            ]

        return render_template(

            "index.html",

            prediction=result,

            confidence=confidence,

            risk=risk,

            recommendation=recommendation,

            model_name="Logistic Regression",

            accuracy="85.37%",

            roc_auc="92.57%"

        )

    except Exception as e:

        return render_template(

            "index.html",

            prediction="Prediction Error",

            confidence=str(e)

        )

# ==========================================
# Run Flask
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)