from flask import Flask, request, jsonify, send_file
import os
import cv2
from werkzeug.utils import secure_filename
from utils import apply_color_overlay

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Helper function to validate file extension
def get_valid_filename(filename):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        ext = '.jpg'  # Default to JPG if extension is invalid or missing
    safe_name = secure_filename(name)
    return f"{safe_name}{ext}"

@app.route('/customize', methods=['POST'])
def customize_image():
    # Check if image is included in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    
    # Get and sanitize the filename
    original_filename = get_valid_filename(img_file.filename)
    img_path = os.path.join(UPLOAD_FOLDER, original_filename)
    img_file.save(img_path)

    # Parse the color from form-data, fallback to red
    color_str = request.form.get('color', '255,0,0')
    try:
        color = tuple(map(int, color_str.split(',')))
        if len(color) != 3:
            raise ValueError
    except ValueError:
        return jsonify({'error': 'Invalid color format. Use R,G,B (e.g., 255,0,0)'}), 400

    # Apply color overlay
    try:
        result_image = apply_color_overlay(img_path, color=color)
    except Exception as e:
        return jsonify({'error': f'Error applying overlay: {str(e)}'}), 500

    # Save the customized image
    output_filename = f"customized_{original_filename}"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    saved = cv2.imwrite(output_path, result_image)

    if not saved:
        return jsonify({'error': 'Failed to save the image. Check extension or OpenCV setup.'}), 500

    # Return the customized image
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
