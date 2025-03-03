import os
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.neighbors import NearestNeighbors
from flask import send_from_directory


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images',filename)
# Load feature embeddings
feature_list = None
filenames = None

try:
    if not os.path.exists('embeddings.pkl') or not os.path.exists('filenames.pkl'):
        raise FileNotFoundError("Missing required files: embeddings.pkl or filenames.pkl")

    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open("filenames.pkl", "rb"))
    

    if feature_list.size == 0 or not isinstance(filenames, list):
        raise ValueError("Feature list or filenames are empty or not in expected format")

    print("Pickle files loaded successfully")

except Exception as e:
    print(f"Error loading pickle files: {e}")
    feature_list = np.array([])  # Initialize as empty array
    filenames = []

# Load model
try:
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    print("Model loaded successfully")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to extract features
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        if model is None:
            raise ValueError("Model is not loaded properly")

        print("Running model prediction...")
        result = model.predict(preprocessed_img).flatten()

        # Normalize features
        normalized_res = result / norm(result, ord=2)
        print("Feature extraction successful")
        return normalized_res

    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

# API Endpoint for Image Upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    print(f"Image received: {img_file.filename}")

    img_path = "static/temp.jpg"
    img_file.save(img_path)

    if not os.path.exists(img_path):
        print("Image not saved properly")
        return jsonify({'error': 'Failed to save image'}), 500

    print("Image saved successfully")
    print(filenames)

    # Extract features
    normalized_res = extract_features(img_path, model)
    if normalized_res is None:
        return jsonify({'error': 'Feature extraction failed'}), 500

    # Ensure feature list is valid
    if feature_list.size == 0 or feature_list is None or not isinstance(feature_list, np.ndarray):
        return jsonify({"error": "Feature extraction data is missing"}), 500

    # Reshape feature list
    reshaped_feature_list = feature_list.reshape(feature_list.shape[0], -1)  # Ensure proper shape

    # Nearest Neighbor Search
    try:
        neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
        neighbors.fit(reshaped_feature_list)
        distances, indices = neighbors.kneighbors([normalized_res])

        matched_images = [f"http://172.16.96.186:5000/images/{os.path.basename(filenames[i])}" for i in indices[0]]
        print(f"Matched images: {matched_images}")

        return jsonify({'matches': matched_images})

    except Exception as e:
        print(f"Error in nearest neighbor search: {e}")
        return jsonify({'error': 'Nearest neighbor search failed'}), 500

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

'''
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False #already trained on imagenet

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D() #adding layer
])
print("holla")
def extract_features(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    prepocessed_img=preprocess_input(expanded_img_array)
    result=model.predict(prepocessed_img).flatten()
    #now need to normalise
    normalized_res = result / norm(result, ord=2)
    return normalized_res
#have to make a list of all the image files
filenames=[]
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))
# now apan ko kya karna h .... sari files ko function m pass karna h zo iske features extract karke de dega
feature_list=[] #2d list hoga of every image features
#and each image will have 2048npm features
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))
pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))
    
    '''