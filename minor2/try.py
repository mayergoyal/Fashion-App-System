import cv2
import face_recognition
import numpy as np

# Load and convert image
image = face_recognition.load_image_file("trial2.jpg")
face_locations = face_recognition.face_locations(image)

if face_locations:
    top, right, bottom, left = face_locations[0]
    face = image[top:bottom, left:right]

    # Convert to HSV for better skin tone detection
    hsv_face = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)

    # Get center part of face (to avoid hair/eyes)
    h, w, _ = hsv_face.shape
    center = hsv_face[h//4:h//2, w//4:w//2]

    avg_color = np.mean(center, axis=(0, 1))  # HSV avg
    print("Average HSV Color (skin tone):", avg_color)

    # Simple skin tone type (light/medium/dark) guess
    v = avg_color[2]
    if v > 180:
        tone = "Light"
    elif v > 100:
        tone = "Medium"
    else:
        tone = "Dark"

    print("Detected Skin Tone:", tone)
else:
    print("No face detected.")
