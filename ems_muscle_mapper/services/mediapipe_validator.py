import os

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    Image as MpImage,
    ImageFormat,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# 3a. MediaPipe Validator


def _create_pose_landmarker():
    """Create a pose landmarker using the MediaPipe Tasks API."""
    model_path = os.environ.get("MEDIA_PIPE_MODEL_PATH") or os.path.join(
        os.path.dirname(__file__), "..", "models", "pose_landmarker_full.task"
    )

    if not os.path.exists(model_path):
        return None

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
    )
    return PoseLandmarker.create_from_options(options)


def verify_arm_presence(image_bytes: bytes) -> bool:
    """Verifies an arm is visible in the image using MediaPipe Pose Tasks."""
    pose_landmarker = None

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_landmarker = _create_pose_landmarker()
        if pose_landmarker is None:
            return False

        mp_image = MpImage(image_format=ImageFormat.SRGB, data=img_rgb)
        results = pose_landmarker.detect(mp_image)
        if not results.pose_landmarks:
            return False

        landmarks = results.pose_landmarks[0].landmark

        # Extract elbow and wrist joints for both arms (MediaPipe Tasks API indices)
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]

        # The joint must have a visibility confidence > 70%
        left_visible = left_elbow.visibility > 0.7 and left_wrist.visibility > 0.7
        right_visible = right_elbow.visibility > 0.7 and right_wrist.visibility > 0.7

        return left_visible or right_visible
    except Exception:
        return False
    finally:
        if pose_landmarker is not None:
            pose_landmarker.close()