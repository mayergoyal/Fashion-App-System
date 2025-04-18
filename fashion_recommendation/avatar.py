import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d

def generate_premium_avatar(image_path, output_path="premium_avatar.png"):
    # Initialize MediaPipe with optimal settings
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    
    # Configurable parameters
    FACE_DETAIL = 3  # 1-5 (higher = more detailed)
    BODY_SMOOTHNESS = 2  # 1-5 (higher = smoother)
    STYLE = "vector"  # "vector" or "realistic"
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose, \
         mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        
        # Image processing
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or invalid path")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        # Create canvas based on style choice
        if STYLE == "vector":
            avatar = np.ones((height, width, 3), dtype=np.uint8) * 255
            line_color = (40, 40, 40)
            fill_color = (240, 240, 240)
        else:
            avatar = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Process pose and face
        pose_results = pose.process(image_rgb)
        face_results = face_mesh.process(image_rgb)
        
        # 1. Enhanced Body Contour (building on your convex hull approach)
        if pose_results.pose_landmarks:
            body_points = []
            for landmark in pose_results.pose_landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)
                body_points.append([x, y])
            
            if len(body_points) > 10:
                # Improved smoothing using spline interpolation
                body_points = np.array(body_points)
                x = body_points[:, 0]
                y = body_points[:, 1]
                
                # Parametric spline interpolation
                t = np.arange(len(x))
                fx = interp1d(t, x, kind='cubic')
                fy = interp1d(t, y, kind='cubic')
                t_new = np.linspace(0, len(x)-1, 100)
                x_new = fx(t_new)
                y_new = fy(t_new)
                
                # Create smoothed contour
                smooth_contour = np.array([x_new, y_new]).T.astype(np.int32)
                
                # Draw body (style-dependent)
                if STYLE == "vector":
                    cv2.polylines(avatar, [smooth_contour], False, line_color, 2)
                    cv2.fillPoly(avatar, [smooth_contour], fill_color)
                else:
                    # Sample skin tone from face
                    skin_color = np.mean(image[max(0,y_new.min()-10):min(height,y_new.min()+10), 
                                        max(0,x_new.min()-10):min(width,x_new.min()+10)], 
                                     axis=(0,1))
                    cv2.fillPoly(avatar, [smooth_contour], skin_color)
        
        # 2. Precision Facial Features (improving your dot approach)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Extract key facial components
                lips = []
                left_eye = []
                right_eye = []
                face_oval = []
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    
                    # Classify landmarks
                    if idx in mp_face_mesh.FACEMESH_LIPS:
                        lips.append((x, y))
                    elif idx in mp_face_mesh.FACEMESH_LEFT_EYE:
                        left_eye.append((x, y))
                    elif idx in mp_face_mesh.FACEMESH_RIGHT_EYE:
                        right_eye.append((x, y))
                    elif idx in mp_face_mesh.FACEMESH_FACE_OVAL:
                        face_oval.append((x, y))
                
                # Draw facial features with precision
                if face_oval:
                    cv2.polylines(avatar, [np.array(face_oval)], True, line_color, 1)
                
                if lips:
                    lip_array = np.array(lips)
                    hull = cv2.convexHull(lip_array)
                    cv2.drawContours(avatar, [hull], 0, (0, 0, 255), -1 if STYLE == "vector" else 1)
                
                if left_eye:
                    eye_array = np.array(left_eye)
                    hull = cv2.convexHull(eye_array)
                    cv2.drawContours(avatar, [hull], 0, (0, 0, 0), -1)
                
                if right_eye:
                    eye_array = np.array(right_eye)
                    hull = cv2.convexHull(eye_array)
                    cv2.drawContours(avatar, [hull], 0, (0, 0, 0), -1)
        
        # 3. Enhanced Output
        cv2.imwrite(output_path, cv2.cvtColor(avatar, cv2.COLOR_RGB2BGR))
        print(f"Premium avatar generated: {output_path}")
        
        # Display
        cv2.imshow("Premium Avatar", avatar)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
generate_premium_avatar("C:\\Users\\Mayer\\Downloads\\m.jpeg", "premium_avatar.png")