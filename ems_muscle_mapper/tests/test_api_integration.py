import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from schemas import MuscleAnalysisResult
from services.arm_region import ArmRegion
from services import vlm_analyzer


JPEG_BYTES = b"\xff\xd8\xfftest-image"
PNG_BYTES = b"\x89PNG\r\n\x1a\ntest-image"


class APIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original_client = vlm_analyzer._client

    def tearDown(self):
        vlm_analyzer._client = self.original_client

    def test_missing_api_key_returns_service_unavailable(self):
        vlm_analyzer._client = None

        def fake_arm_region(image_bytes, preferred_side=None):
            return ArmRegion(
                image_bytes=image_bytes,
                side=preferred_side or "left",
                left=0,
                top=0,
                right=100,
                bottom=100,
                source_width=100,
                source_height=100,
            )

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(main, "normalize_image_orientation", side_effect=lambda value: value),
            patch.object(main, "extract_arm_region", side_effect=fake_arm_region),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            response = TestClient(main.app).post(
                "/analyze",
                files={
                    "lax_image": ("lax.jpg", JPEG_BYTES, "image/jpeg"),
                    "flexed_image": ("flexed.jpg", JPEG_BYTES, "image/jpeg"),
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENAI_API_KEY", response.json()["detail"])

    def test_success_returns_image_and_alt_text_json(self):
        analysis = MuscleAnalysisResult.model_validate(
            {
                "movement_detected": "Elbow flexion",
                "muscles": [
                    {
                        "name": "Biceps",
                        "polygon_vertices_normalized": [
                            {"x": 0.1, "y": 0.2},
                            {"x": 0.2, "y": 0.2},
                            {"x": 0.2, "y": 0.4},
                        ],
                        "color_hex": "#ff0000",
                        "ems_pads_normalized": [
                            {"label": "Proximal", "x": 0.15, "y": 0.25}
                        ],
                    }
                ],
            }
        )

        def fake_arm_region(image_bytes, preferred_side=None):
            return ArmRegion(
                image_bytes=image_bytes,
                side=preferred_side or "left",
                left=0,
                top=0,
                right=100,
                bottom=100,
                source_width=100,
                source_height=100,
            )

        with (
            patch.object(main, "normalize_image_orientation", side_effect=lambda value: value),
            patch.object(main, "extract_arm_region", side_effect=fake_arm_region),
            patch.object(main, "analyze_muscle_movement", return_value=analysis),
            patch.object(main, "draw_ems_ui", return_value=b"annotated-jpeg"),
        ):
            response = TestClient(main.app).post(
                "/analyze",
                files={
                    "lax_image": ("lax.jpg", JPEG_BYTES, "image/jpeg"),
                    "flexed_image": ("flexed.jpg", JPEG_BYTES, "image/jpeg"),
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["image_base64"], "YW5ub3RhdGVkLWpwZWc=")
        self.assertIn("Elbow flexion", payload["alt_text"])
        self.assertIn("Biceps", payload["alt_text"])

    def test_analyzer_uses_each_image_actual_media_type(self):
        parsed = MuscleAnalysisResult(movement_detected="Elbow flexion", muscles=[])
        captured = {}

        def fake_parse(**kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(parsed=parsed, refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        vlm_analyzer._client = SimpleNamespace(
            beta=SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(parse=fake_parse),
                )
            )
        )

        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o"}):
            result = vlm_analyzer.analyze_muscle_movement(PNG_BYTES, JPEG_BYTES)

        content = captured["messages"][0]["content"]
        self.assertTrue(
            content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        )
        self.assertTrue(
            content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )
        self.assertEqual(captured["model"], "gpt-4o")
        self.assertEqual(content[1]["image_url"]["detail"], "high")
        self.assertEqual(content[2]["image_url"]["detail"], "high")
        self.assertIs(result, parsed)

    def test_api_key_is_read_from_the_environment(self):
        sentinel_client = object()

        with (
            patch.object(vlm_analyzer, "OpenAI", return_value=sentinel_client) as constructor,
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "test-key-from-environment"},
                clear=False,
            ),
        ):
            os.environ.pop("OPENAI_BASE_URL", None)
            vlm_analyzer._client = None
            self.assertIs(vlm_analyzer._get_client(), sentinel_client)

        constructor.assert_called_once_with(
            api_key="test-key-from-environment",
            base_url="https://api.openai.com/v1",
            timeout=60.0,
            max_retries=2,
        )

    def test_blank_base_url_uses_openai_default(self):
        sentinel_client = object()

        with (
            patch.object(vlm_analyzer, "OpenAI", return_value=sentinel_client) as constructor,
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key-from-environment",
                    "OPENAI_BASE_URL": "",
                },
                clear=False,
            ),
        ):
            vlm_analyzer._client = None
            self.assertIs(vlm_analyzer._get_client(), sentinel_client)

        constructor.assert_called_once_with(
            api_key="test-key-from-environment",
            base_url="https://api.openai.com/v1",
            timeout=60.0,
            max_retries=2,
        )


if __name__ == "__main__":
    unittest.main()
