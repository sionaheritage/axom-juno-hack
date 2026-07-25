from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from schemas import MuscleAnalysisResult
from services.arm_region import ArmRegion, extract_arm_region
from services.image_processor import build_alt_text
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


if __name__ == "__main__":
    unittest.main()
