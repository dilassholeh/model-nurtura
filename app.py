from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# load model & scaler
model = joblib.load("model_kmeans.pkl")
scaler = joblib.load("scaler.pkl")


@app.route('/')
def home():
    return jsonify({
        "message": "API ML aktif"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON received"
            }), 400

        features = data.get("features")

        if not features:
            return jsonify({
                "status": "error",
                "message": "Features is required"
            }), 400

        print("INPUT:", features)

        input_arr = np.array(features).reshape(1, -1)

        print("ARRAY:", input_arr)

        input_scaled = scaler.transform(input_arr)

        print("SCALED:", input_scaled)

        cluster = model.predict(input_scaled)[0]

        print("PREDIKSI CLUSTER:", cluster)

        if cluster == 0:
            result = "Beresiko"
        else:
            result = "Tidak beresiko"

        return jsonify({
            "status": "success",
            "cluster": int(cluster),
            "result": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port
    )