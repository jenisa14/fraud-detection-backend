from flask import Flask, request, jsonify
from flask_cors import CORS  # This is crucial for React integration
import joblib
import pandas as pd

app = Flask(__name__)
# Allow React (running on any port) to access this API
CORS(app) 

print("⏳ Loading models...")
try:
    # Load your actual trained files
    model_columns = joblib.load("model_columns.pkl")
    model = joblib.load("fraud_detection_model.pkl") 
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Make sure .pkl files are in the same folder as app.py")

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "Backend running",
        "model_loaded": True
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from frontend
        json_input = request.json
        
        # 2. Convert to DataFrame
        query_df = pd.DataFrame([json_input])
        
        # 3. Realign columns to match training data
        # (This handles missing columns or different ordering)
        query = pd.get_dummies(query_df)
        query = query.reindex(columns=model_columns, fill_value=0)
        
        # 4. Make prediction
        prediction = model.predict(query)
        probability = model.predict_proba(query)
        
        # 5. Send result back to React
        return jsonify({
            "success": True,
            "prediction": "Fraud" if prediction[0] == 1 else "Not Fraud",
            "probability": float(probability[0][1]),
            "model_used": "L1 LASSO"
        })
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
