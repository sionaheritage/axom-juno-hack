from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from schemas import MuscleAnalysisResult
from services.arm_region import ArmRegion, _infer_flexion_side, extract_arm_region
from services.image_processor import (
    _boxes_overlap,
    _choose_label_position,
    build_alt_text,
)
from services.image_normalizer import normalize_image_orientation


class _ArrayWrapper:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class SpatialPipelineTests(unittest.TestCase):
    def test_exif_orientation_is_applied_and_removed(self):
        source = Image.new("RGB", (40, 20), "white")
        exif = Image.Exif()
        exif[274] = 6  # Rotate 90 degrees clockwise for display.
        encoded = BytesIO()
        source.save(encoded, format="JPEG", exif=exif)

        normalized = normalize_image_orientation(encoded.getvalue())

        with Image.open(BytesIO(normalized)) as result:
            self.assertEqual(result.size, (20, 40))
            self.assertIsNone(result.getexif().get(274))

    def test_yolo_points_create_a_left_side_crop(self):
        image = np.zeros((120, 240, 3), dtype=np.uint8)
        encoded, buffer = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)

        keypoints = np.zeros((1, 17, 3), dtype=np.float32)
        keypoints[0, 5] = [30, 30, 0.95]  # Left shoulder
        keypoints[0, 7] = [45, 55, 0.95]  # Left elbow
        keypoints[0, 9] = [55, 90, 0.95]  # Left wrist
        fake_result = SimpleNamespace(
            keypoints=SimpleNamespace(data=_ArrayWrapper(keypoints))
        )

        with patch("services.arm_region.pose_model", return_value=[fake_result]):
            region = extract_arm_region(buffer.tobytes())

        self.assertEqual(region.side, "left")
        self.assertLess(region.right, image.shape[1] // 2)
        self.assertEqual(region.source_width, 240)
        self.assertEqual(region.source_height, 120)
        self.assertIsNotNone(region.shoulder_crop)
        self.assertIsNotNone(region.elbow_crop)
        self.assertIsNotNone(region.wrist_crop)
        self.assertIn("subject's left arm", region.pose_prompt_context())

    def test_demo_pose_places_biceps_on_right_side_of_crop(self):
        shoulder = np.array([288.6, 478.6])
        elbow = np.array([404.1, 1534.3])
        wrist = np.array([1092.8, 1212.3])

        self.assertEqual(_infer_flexion_side(shoulder, elbow, wrist), "right")

    def test_refinement_enlarges_shape_and_corrects_reversed_labels(self):
        region = ArmRegion(
            image_bytes=b"crop",
            side="right",
            left=0,
            top=0,
            right=100,
            bottom=100,
            source_width=100,
            source_height=100,
            shoulder_crop=(0.5, 0.1),
            elbow_crop=(0.5, 0.9),
            wrist_crop=(1.2, 0.7),
            flexion_side="right",
        )
        analysis = MuscleAnalysisResult.model_validate(
            {
                "movement_detected": "Elbow flexion",
                "muscles": [
                    {
                        "name": "Biceps",
                        "polygon_vertices_normalized": [
                            {"x": 0.25, "y": 0.4},
                            {"x": 0.35, "y": 0.4},
                            {"x": 0.35, "y": 0.6},
                            {"x": 0.25, "y": 0.6},
                        ],
                        "color_hex": "#ff0000",
                        "ems_pads_normalized": [],
                    },
                    {
                        "name": "Triceps",
                        "polygon_vertices_normalized": [
                            {"x": 0.65, "y": 0.4},
                            {"x": 0.75, "y": 0.4},
                            {"x": 0.75, "y": 0.6},
                            {"x": 0.65, "y": 0.6},
                        ],
                        "color_hex": "#00ff00",
                        "ems_pads_normalized": [],
                    },
                ],
            }
        )

        refined = region.refine_crop_analysis(analysis)

        self.assertEqual(refined.muscles[0].name, "Triceps")
        self.assertEqual(refined.muscles[1].name, "Biceps")
        for muscle in refined.muscles:
            xs = [point.x for point in muscle.polygon_vertices_normalized]
            ys = [point.y for point in muscle.polygon_vertices_normalized]
            self.assertGreaterEqual((max(xs) - min(xs)) * (max(ys) - min(ys)), 0.079)
        # Refinement does not mutate the raw API result.
        self.assertEqual(analysis.muscles[0].name, "Biceps")

    def test_crop_coordinates_map_back_to_source(self):
        crop = ArmRegion(
            image_bytes=b"crop",
            side="left",
            left=10,
            top=20,
            right=60,
            bottom=60,
            source_width=200,
            source_height=100,
        )
        analysis = MuscleAnalysisResult.model_validate(
            {
                "movement_detected": "test",
                "muscles": [
                    {
                        "name": "test",
                        "polygon_vertices_normalized": [
                            {"x": 0.0, "y": 0.0},
                            {"x": 0.5, "y": 0.5},
                            {"x": 1.0, "y": 1.0},
                        ],
                        "color_hex": "#ff0000",
                        "ems_pads_normalized": [
                            {"label": "pad", "x": 0.5, "y": 0.5},
                        ],
                    }
                ],
            }
        )

        mapped = crop.map_analysis_to_source(analysis)
        top_left, center, bottom_right = (
            mapped.muscles[0].polygon_vertices_normalized
        )
        pad = mapped.muscles[0].ems_pads_normalized[0]

        self.assertAlmostEqual(top_left.x, 0.05)
        self.assertAlmostEqual(top_left.y, 0.2)
        self.assertAlmostEqual(center.x, 0.175)
        self.assertAlmostEqual(center.y, 0.4)
        self.assertAlmostEqual(bottom_right.x, 0.3)
        self.assertAlmostEqual(bottom_right.y, 0.6)
        self.assertAlmostEqual(pad.x, 0.175)
        self.assertAlmostEqual(pad.y, 0.4)
        # Mapping is non-mutating.
        self.assertEqual(analysis.muscles[0].polygon_vertices_normalized[0].x, 0.0)

    def test_alt_text_describes_movement_muscle_and_pads(self):
        analysis = MuscleAnalysisResult.model_validate(
            {
                "movement_detected": "Bicep flexing",
                "muscles": [
                    {
                        "name": "Biceps",
                        "polygon_vertices_normalized": [
                            {"x": 0.1, "y": 0.2},
                        ],
                        "color_hex": "#ff0000",
                        "ems_pads_normalized": [
                            {"label": "Proximal", "x": 0.1, "y": 0.2},
                            {"label": "Distal", "x": 0.2, "y": 0.3},
                        ],
                    }
                ],
            }
        )

        description = build_alt_text(analysis)
        self.assertIn("Bicep flexing", description)
        self.assertIn("Biceps", description)
        self.assertIn("Proximal, Distal", description)

    def test_all_pointer_labels_receive_non_overlapping_boxes(self):
        pointer = (250, 200)
        occupied = []
        origins = []
        for label in ["Biceps", "Triceps", "Proximal", "Distal"]:
            origin, bounds = _choose_label_position(
                label, pointer, 1.0, 600, 500, occupied
            )
            self.assertFalse(
                any(_boxes_overlap(bounds, previous) for previous in occupied)
            )
            occupied.append(bounds)
            origins.append(origin)

        self.assertTrue(any(origin[0] < pointer[0] for origin in origins))


if __name__ == "__main__":
    unittest.main()
