from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import cv2

app = Flask(__name__)

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

        return jsonify(clean_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
