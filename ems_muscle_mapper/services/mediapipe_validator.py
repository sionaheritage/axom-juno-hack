import cv2
import mediapipe as mp
import numpy as np

# 3a. MediaPipe Validator

mp_pose = mp.solutions.pose

def verify_arm_presence(image_bytes: bytes) -> bool:
    """Verifies an arm is visible in the image using MediaPipe Pose."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(img_rgb)
        if not results.pose_landmarks:
            return False
            
        landmarks = results.pose_landmarks.landmark
        
        # Extract elbow and wrist joints for both arms
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # The joint must have a visibility confidence > 70%
        left_visible = left_elbow.visibility > 0.7 and left_wrist.visibility > 0.7
        right_visible = right_elbow.visibility > 0.7 and right_wrist.visibility > 0.7
        
        return left_visible or right_visible