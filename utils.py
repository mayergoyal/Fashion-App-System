import cv2
import numpy as np
import mediapipe as mp

def apply_color_overlay(img_path, color=(0, 255, 0)):
    # Convert RGB to BGR for OpenCV
    color = color[::-1]  # RGB to BGR

    # Read and convert image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe SelfieSegmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
        results = selfie_seg.process(img_rgb)

        # Get segmentation mask
        mask = results.segmentation_mask
        condition = mask > 0.6  # Threshold to define "person" area

        # Create a solid color overlay
        color_bkg = np.zeros_like(img, dtype=np.uint8)
        color_bkg[:] = color

        # Apply overlay where mask is True
        blended = np.where(condition[..., np.newaxis], color_bkg, img)

        # Blend with original image
        output = cv2.addWeighted(blended, 0.6, img, 0.4, 0)

    return output
