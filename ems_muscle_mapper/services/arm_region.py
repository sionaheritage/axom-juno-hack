from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from schemas import MuscleAnalysisResult
from services.arm_validator import pose_model


ArmSide = Literal["left", "right"]
ImageSide = Literal["left", "right", "uncertain"]


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
    shoulder_crop: tuple[float, float] | None = None
    elbow_crop: tuple[float, float] | None = None
    wrist_crop: tuple[float, float] | None = None
    flexion_side: ImageSide = "uncertain"

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

    def pose_prompt_context(self) -> str:
        """Describe the detected anatomy in crop-normalized coordinates."""
        if not self.shoulder_crop or not self.elbow_crop or not self.wrist_crop:
            return (
                f"YOLO identifies this as the subject's {self.side} arm. "
                "Detailed joint geometry is unavailable."
            )

        context = (
            f"YOLO identifies this as the subject's {self.side} arm. "
            f"In second-crop coordinates, shoulder={_format_point(self.shoulder_crop)}, "
            f"elbow={_format_point(self.elbow_crop)}, and "
            f"wrist={_format_point(self.wrist_crop)}. "
            "The wrist may lie outside the upper-arm crop."
        )
        if self.flexion_side != "uncertain":
            context += (
                f" The forearm bends toward the {self.flexion_side} side of the "
                "crop. For elbow flexion, biceps belongs on that flexion/anterior "
                "side and triceps on the opposite side. Do not reverse them."
            )
        return context

    def refine_crop_analysis(
        self, analysis: MuscleAnalysisResult
    ) -> MuscleAnalysisResult:
        """
        Preserve model contours while correcting labels and undersized polygons.

        The VLM remains responsible for the actual contour. Small center-biased
        polygons are expanded around their own centroid, retaining their vertex
        order and shape instead of replacing them with a generic rectangle.
        """
        refined = analysis.model_copy(deep=True)
        self._correct_biceps_triceps_labels(refined)

        for muscle in refined.muscles:
            _expand_small_polygon(muscle.polygon_vertices_normalized)
        return refined

    def _correct_biceps_triceps_labels(
        self, analysis: MuscleAnalysisResult
    ) -> None:
        if (
            self.flexion_side == "uncertain"
            or not self.shoulder_crop
            or not self.elbow_crop
        ):
            return

        biceps = next(
            (muscle for muscle in analysis.muscles if "bicep" in muscle.name.lower()),
            None,
        )
        triceps = next(
            (muscle for muscle in analysis.muscles if "tricep" in muscle.name.lower()),
            None,
        )
        if not biceps or not triceps:
            return

        biceps_score = self._lateral_score(biceps.polygon_vertices_normalized)
        triceps_score = self._lateral_score(triceps.polygon_vertices_normalized)
        if biceps_score is None or triceps_score is None:
            return

        # Positive score means crop-right. Biceps must be on the flexion side.
        desired_sign = 1.0 if self.flexion_side == "right" else -1.0
        if biceps_score * desired_sign < triceps_score * desired_sign:
            biceps.name, triceps.name = triceps.name, biceps.name

    def _lateral_score(self, points) -> float | None:
        if not points or not self.shoulder_crop or not self.elbow_crop:
            return None

        shoulder = np.array(self.shoulder_crop, dtype=float)
        elbow = np.array(self.elbow_crop, dtype=float)
        axis = elbow - shoulder
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            return None

        centroid = np.mean([[point.x, point.y] for point in points], axis=0)
        midpoint = (shoulder + elbow) / 2.0
        crop_right_normal = np.array([axis[1], -axis[0]]) / length
        return float(np.dot(centroid - midpoint, crop_right_normal))


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

    shoulder, elbow, wrist = points
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

    def to_crop_coordinates(point: np.ndarray) -> tuple[float, float]:
        return (
            float((point[0] - left) / (right - left)),
            float((point[1] - top) / (bottom - top)),
        )

    return ArmRegion(
        image_bytes=buffer.tobytes(),
        side=side,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        source_width=source_width,
        source_height=source_height,
        shoulder_crop=to_crop_coordinates(shoulder),
        elbow_crop=to_crop_coordinates(elbow),
        wrist_crop=to_crop_coordinates(wrist),
        flexion_side=_infer_flexion_side(shoulder, elbow, wrist),
    )


def _expand_small_polygon(
    points,
    minimum_bbox_area: float = 0.08,
    minimum_major_span: float = 0.45,
    maximum_scale: float = 2.2,
) -> None:
    """Expand only undersized polygons, preserving their contour and centroid."""
    if len(points) < 3:
        return

    coordinates = np.array([[point.x, point.y] for point in points], dtype=float)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    spans = maximum - minimum
    bbox_area = float(spans[0] * spans[1])
    major_span = float(max(spans))

    area_scale = (
        np.sqrt(minimum_bbox_area / bbox_area) if bbox_area > 1e-8 else maximum_scale
    )
    span_scale = (
        minimum_major_span / major_span if major_span > 1e-8 else maximum_scale
    )
    scale = min(maximum_scale, max(1.0, area_scale, span_scale))
    if scale <= 1.0:
        return

    centroid = coordinates.mean(axis=0)
    expanded = centroid + (coordinates - centroid) * scale
    expanded = np.clip(expanded, 0.01, 0.99)
    for point, (x, y) in zip(points, expanded):
        point.x = float(x)
        point.y = float(y)


def _infer_flexion_side(
    shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray
) -> ImageSide:
    upper_arm = elbow - shoulder
    length = float(np.linalg.norm(upper_arm))
    if length < 1.0:
        return "uncertain"

    midpoint = (shoulder + elbow) / 2.0
    image_right_normal = np.array([upper_arm[1], -upper_arm[0]])
    signed_distance = float(np.dot(wrist - midpoint, image_right_normal) / length)
    if abs(signed_distance) < max(5.0, length * 0.03):
        return "uncertain"
    return "right" if signed_distance > 0 else "left"


def _format_point(point: tuple[float, float]) -> str:
    return f"({point[0]:.3f}, {point[1]:.3f})"


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
