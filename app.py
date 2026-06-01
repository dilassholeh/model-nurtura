from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

from services.recommendation_service import get_ai_recommendation
from services.chatbot_service import get_chatbot_reply
from utils.screening_features import build_features

app = Flask(__name__)

# load model & scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
print("MODEL =", model)
print("CENTROIDS =")
print(scaler.inverse_transform(model.cluster_centers_))
CLUSTER_BERESIKO = 0

# MongoDB connection
mongo_uri = os.environ.get("MONGODB_URI", "mongodb+srv://userNurtura:nurturame123@cluster0.2vph2f5.mongodb.net/DBnurtura?authSource=admin")
client = MongoClient(mongo_uri)
db = client['DBnurtura']
users_collection = db['users']


@app.route('/predict', methods=['POST'])
def predict():
    try:
    
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "JSON tidak ditemukan"
            }), 400

        answers = data.get("answers")
        mother_id = data.get("mother_id")

        if answers is None or mother_id is None:
            return jsonify({
                "status": "error",
                "message": "answers atau mother_id tidak ditemukan"
            }), 400
        
        # Validasi mother_id ada di collection users
        try:
            print(f"DEBUG: Checking mother_id: {mother_id}")
            mother_obj_id = ObjectId(mother_id)
            print(f"DEBUG: Created ObjectId: {mother_obj_id}")
            
            mother = users_collection.find_one({"_id": mother_obj_id})
            print(f"DEBUG: Found mother: {mother is not None}")
            
            if not mother:
                print(f"DEBUG: Mother not found, returning 404")
                return jsonify({
                    "status": "error",
                    "message": "Mother tidak ditemukan dalam sistem"
                }), 404
            
            # Validasi mother memiliki role 'mother'
            mother_role = mother.get('role')
            print(f"DEBUG: Mother role: {mother_role}")
            if mother_role != 'mother':
                return jsonify({
                    "status": "error",
                    "message": "User bukan mother"
                }), 403
        except Exception as e:
            print(f"DEBUG: Exception during validation: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Invalid mother_id: {str(e)}"
            }), 400

        # Map answers to features array based on ml_index
        features = build_features(answers)

        # ubah ke numpy array
        input_arr = np.array(features).reshape(1, -1)

        # scaling
        input_scaled = scaler.transform(input_arr)

        # prediksi
        cluster = model.predict(input_scaled)[0]

        # mapping hasil
        result = "Beresiko Depresi" if cluster == CLUSTER_BERESIKO else "Tidak Beresiko Depresi"

        recommendation = get_ai_recommendation(result, answers, features, cluster)

        return jsonify({
            "status": "success",
            "cluster": int(cluster),
            "result": result,
            "recommendation": recommendation
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok"
    })


@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "JSON tidak ditemukan"
            }), 400

        message = str(data.get("message", "")).strip()
        user_role = str(data.get("user_role", "")).strip().lower()

        if not message:
            return jsonify({
                "status": "error",
                "message": "message wajib diisi"
            }), 400

        if user_role not in ["mother", "father"]:
            return jsonify({
                "status": "error",
                "message": "user_role harus mother atau father"
            }), 400

        context = data.get("context", {})
        if not isinstance(context, dict) or not context.get("latest_prediction"):
            return jsonify({
                "status": "error",
                "message": "latest_prediction wajib tersedia untuk menggunakan chatbot"
            }), 400

        reply = get_chatbot_reply(
            message,
            user_role,
            context,
            data.get("history", [])
        )

        return jsonify({
            "status": "success",
            **reply
        })
    except Exception as e:
        print(f"Chatbot error: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

print("\n=== ROUTES ===")
print(app.url_map)
print("==============\n")

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port
    )