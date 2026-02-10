from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

print("⏳ Loading models...")
try:
    model_columns = joblib.load("model_columns.pkl")
    model = joblib.load("fraud_detection_model.pkl")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "Backend running",
        "model_loaded": True
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_input = request.json
        query_df = pd.DataFrame([json_input])
        query = pd.get_dummies(query_df)
        query = query.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(query)
        probability = model.predict_proba(query)

        return jsonify({
            "success": True,
            "prediction": "Fraud" if prediction[0] == 1 else "Not Fraud",
            "probability": float(probability[0][1])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
