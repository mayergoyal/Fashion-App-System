from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({"message": "Server is running"})

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(img_path=img, actions=['age', 'gender', 'race'], enforce_detection=False)

        def convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        clean_result = convert(result)

        # Extract traits
        gender = clean_result[0]['dominant_gender']
        age = clean_result[0]['age']
        race = clean_result[0]['dominant_race']

        # 👕 Clothing recommendation logic
        recs = []

        if gender == "Man":
            recs.append("Slim-fit shirts")
            recs.append("Casual blazers")
            recs.append("Dark jeans")
        else:
            recs.append("Floral tops")
            recs.append("Denim jackets")
            recs.append("Chic dresses")

        if age < 25:
            recs.append("Trendy streetwear")
        elif age < 40:
            recs.append("Smart casuals")
        else:
            recs.append("Elegant classics")

        # Optional race-based color/style suggestion
        if race == "asian":
            recs.append("Pastel shades")
        elif race == "black":
            recs.append("Bold prints")
        elif race == "white":
            recs.append("Earth tones")

        return jsonify({
            "analysis": clean_result[0],
            "recommendations": recs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port=5001)

