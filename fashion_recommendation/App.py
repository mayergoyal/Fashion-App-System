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
#for segmentation 

import cv2
import torch
import segmentation_models_pytorch as smp

def segment_garment(garment_path):
    #loading pretrained u net model 
    model=smp.Unet('resnet34',encoder_weights='imagenet',classes=1,activation='sigmoid')
    model.load_state_dict(torch.load('unet_garment_segmentation.pth'))
    model.eval()
    
    # loading and preprocessing the garment image 
    image=cv2.imread(garment_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))  # Resize to model input size
    image = image / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        mask=model(image)
        mask=(mask>.5).float()
        
    mask=mask.squeeze().numpy()
    mask=(mask*255).astype(np.uint8)
    mask=cv2.resize(mask,(image.shape[2],image.shape[1]))
    return mask

#pose estimation 
import mediapipe as mp


def estimate_pose(user_path):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

    # Load the user photo
    image = cv2.imread(user_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(image)

    # Extract keypoints
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append((landmark.x, landmark.y))  # Normalized coordinates (0-1)

    pose.close()
    return keypoints

#warp garment
from scipy.interpolate import Rbf

def warp_garment(garment_path, garment_mask, pose_keypoints):
    # Load the garment image and mask
    garment = cv2.imread(garment_path)
    garment = cv2.cvtColor(garment, cv2.COLOR_BGR2RGB)
    garment_mask = cv2.resize(garment_mask, (garment.shape[1], garment.shape[0]))

    # Define source points (garment corners)
    h, w = garment.shape[:2]
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Define target points (based on pose keypoints)
    # Example: Use shoulder and hip keypoints
    target_points = np.float32([
        [pose_keypoints[11][0] * w, pose_keypoints[11][1] * h],  # Left shoulder
        [pose_keypoints[12][0] * w, pose_keypoints[12][1] * h],  # Right shoulder
        [pose_keypoints[23][0] * w, pose_keypoints[23][1] * h],  # Left hip
        [pose_keypoints[24][0] * w, pose_keypoints[24][1] * h]   # Right hip
    ])

    # Compute TPS transformation
    tps = Rbf(src_points[:, 0], src_points[:, 1], target_points[:, 0], target_points[:, 1], function='thin_plate')
    warped_garment = cv2.remap(garment, tps(src_points[:, 0], tps(src_points[:, 1]), interpolation=cv2.INTER_LINEAR))

    return warped_garment
    
@app.route('/virtual-try-on',methods=['POST'])
def virtual_try_on():
    if 'user_photo' not in request.files or 'garment_photo' not in request.files:
        return jsonify({'error':'Both photos of user and garment are required'}),400
    user_photo=request.files['user_photo']
    garment_photo=request.files['garment_photo']
    
    #pehle toh save karte hain uploaded photos 
    user_photo_path='static/user_photo.jpg'
    garment_photo_path='static/garment_photo.jpg'
    user_photo.save(user_photo_path)
    garment_photo.save(garment_photo_path)
    
    # now time for performing virtual trial wali cheez 
    try:
        #sabse pehle segment karna h matlab user ki body ko segment karna and garment ko bhi so that overlay ho ske realistically
        garment_mask=segment_garment(garment_photo_path)
        # now estimate the user's pose 
        pose_keypoints=estimate_pose(user_photo_path)
        #warping the garment to fit the user;'s pose
        warped_garment=warp_garment(garment_photo_path,garment_mask,warped_garment)
        #finally blend karenge 
        res_image=blend_image(user_photo_path,warped_garment)
        
        result_path='static/result.jpg'
        cv2.imwrite(result_path,res_image)
        return jsonify({'result': f"http://192.168.109.188:5000/static/result.jpg"})
    except Exception as e:
        print(f"Error in virtual try-on: {e}")
        return jsonify({'error': 'Virtual try-on failed'}), 500
        
    
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

        matched_images = [f"http://192.168.109.188:5000/images/{os.path.basename(filenames[i])}" for i in indices[0]]
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