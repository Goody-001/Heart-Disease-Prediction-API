from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return {"message": "Heart Disease Prediction API running!"}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input
    # Example: {"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,
    #           "thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}

    features = np.array([list(data.values())]).reshape(1, -1)
    prediction = model.predict(features)[0]

    return jsonify({"prediction": int(prediction)})
