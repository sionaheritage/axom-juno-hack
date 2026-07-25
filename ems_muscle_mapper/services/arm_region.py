from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from schemas import MuscleAnalysisResult
from services.arm_validator import pose_model


ArmSide = Literal["left", "right"]


class ArmNotFoundError(ValueError):
    """Raised when shoulder, elbow, and wrist cannot define a reliable arm crop."""


@dataclass(frozen=True)
class ArmRegion:
    image_bytes: bytes
    side: ArmSide
    left: int
    top: int
    right: int
    bottom: int
    source_width: int
    source_height: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def map_analysis_to_source(
        self, analysis: MuscleAnalysisResult
    ) -> MuscleAnalysisResult:
        """Map crop-normalized coordinates back to normalized source coordinates."""
        mapped = analysis.model_copy(deep=True)

        def map_x(value: float) -> float:
            crop_x = min(1.0, max(0.0, value))
            return min(
                1.0,
                max(0.0, (self.left + crop_x * self.width) / self.source_width),
            )

        def map_y(value: float) -> float:
            crop_y = min(1.0, max(0.0, value))
            return min(
                1.0,
                max(0.0, (self.top + crop_y * self.height) / self.source_height),
            )

        for muscle in mapped.muscles:
            for point in muscle.polygon_vertices_normalized:
                point.x = map_x(point.x)
                point.y = map_y(point.y)
            for pad in muscle.ems_pads_normalized:
                pad.x = map_x(pad.x)
                pad.y = map_y(pad.y)

        return mapped


def extract_arm_region(
    image_bytes: bytes,
    preferred_side: ArmSide | None = None,
    confidence_threshold: float = 0.5,
) -> ArmRegion:
    """
    Crop the upper-arm region from the best shoulder-elbow-wrist detection.

    Wrist confidence is still required to establish a complete, unambiguous arm,
    but the VLM crop is constrained to shoulder-to-elbow where the primary
    elbow-flexion muscles and EMS pads are located.
    """
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ArmNotFoundError("The normalized image could not be decoded.")

    results = pose_model(image, verbose=False)
    if not results or results[0].keypoints is None:
        raise ArmNotFoundError(_arm_error_message(preferred_side))

    keypoints_data = results[0].keypoints.data.cpu().numpy()
    if len(keypoints_data) == 0:
        raise ArmNotFoundError(_arm_error_message(preferred_side))

    # COCO indices: left shoulder/elbow/wrist = 5/7/9,
    # right shoulder/elbow/wrist = 6/8/10.
    definitions: dict[ArmSide, tuple[int, int, int]] = {
        "left": (5, 7, 9),
        "right": (6, 8, 10),
    }
    sides = [preferred_side] if preferred_side else ["left", "right"]
    candidates = []

    for person_keypoints in keypoints_data:
        for side in sides:
            indices = definitions[side]
            points = person_keypoints[list(indices)]
            confidences = points[:, 2]
            if np.all(confidences >= confidence_threshold):
                candidates.append((float(np.mean(confidences)), side, points[:, :2]))

    if not candidates:
        raise ArmNotFoundError(_arm_error_message(preferred_side))

    _, side, points = max(candidates, key=lambda candidate: candidate[0])
    source_height, source_width = image.shape[:2]

    shoulder, elbow, _wrist = points
    crop_points = np.array([shoulder, elbow])
    upper_arm_length = float(np.linalg.norm(shoulder - elbow))
    padding = max(24, int(round(upper_arm_length * 0.22)))

    left = max(0, int(np.floor(np.min(crop_points[:, 0]))) - padding)
    top = max(0, int(np.floor(np.min(crop_points[:, 1]))) - padding)
    right = min(
        source_width, int(np.ceil(np.max(crop_points[:, 0]))) + padding + 1
    )
    bottom = min(
        source_height, int(np.ceil(np.max(crop_points[:, 1]))) + padding + 1
    )

    if right - left < 2 or bottom - top < 2:
        raise ArmNotFoundError(_arm_error_message(preferred_side))

    crop = image[top:bottom, left:right]
    encoded, buffer = cv2.imencode(
        ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    if not encoded:
        raise ArmNotFoundError("The detected arm region could not be encoded.")

    return ArmRegion(
        image_bytes=buffer.tobytes(),
        side=side,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        source_width=source_width,
        source_height=source_height,
    )


def _arm_error_message(preferred_side: ArmSide | None) -> str:
    if preferred_side:
        return (
            f"Could not detect the same {preferred_side} shoulder, elbow, and wrist "
            "in both images."
        )
    return (
        "Could not detect a clear shoulder, elbow, and wrist. "
        "Ensure one complete arm is visible."
    )
