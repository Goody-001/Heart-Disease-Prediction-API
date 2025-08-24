from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained model (make sure heart_disease_model.pkl exists in your repo)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Heart Disease Prediction API running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Example expected JSON:
        # {
        #   "age": 52,
        #   "sex": 1,
        #   "cp": 0,
        #   "trestbps": 125,
        #   "chol": 212,
        #   "fbs": 0,
        #   "restecg": 1,
        #   "thalach": 168,
        #   "exang": 0,
        #   "oldpeak": 1.0,
        #   "slope": 2,
        #   "ca": 0,
        #   "thal": 2
        # }

        # Convert JSON values into a numpy array (list of features)
        features = [
            data["age"], data["sex"], data["cp"], data["trestbps"], data["chol"],
            data["fbs"], data["restecg"], data["thalach"], data["exang"],
            data["oldpeak"], data["slope"], data["ca"], data["thal"]
        ]

        # Reshape for prediction (1 sample, 13 features)
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)[0]

        # Return result
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return jsonify({"prediction": int(prediction), "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Run locally (Render will handle Gunicorn in production)
    app.run(host="0.0.0.0", port=5000)
