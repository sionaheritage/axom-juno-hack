import os
import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from schemas import MuscleAnalysisResult

# Load local development settings without depending on the process working directory.
# The project-specific file intentionally overrides stale shell/user variables.
# In production (where this file is absent), the hosting environment is still used.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)


class APIConfigurationError(RuntimeError):
    """Raised when the server has not been configured for the VLM provider."""


class UnsupportedImageError(ValueError):
    """Raised when an uploaded image format cannot be sent to the VLM."""


_client = None


def _get_client() -> OpenAI:
    """Create the API client lazily so a missing key does not crash the server."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise APIConfigurationError(
            "OPENAI_API_KEY is not configured on the server. "
            "Set it in ems_muscle_mapper/.env for local development."
        )

    client_options = {
        "api_key": api_key,
        "base_url": os.getenv("OPENAI_BASE_URL", "").strip()
        or "https://api.openai.com/v1",
        "timeout": 60.0,
        "max_retries": 2,
    }

    _client = OpenAI(**client_options)
    return _client


def _image_media_type(image_bytes: bytes) -> str:
    """Return a data-URL media type supported by OpenAI image inputs."""
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if (
        len(image_bytes) >= 12
        and image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    raise UnsupportedImageError(
        "Unsupported image format. Upload a JPEG, PNG, GIF, or WebP image."
    )

def analyze_muscle_movement(
    lax_bytes: bytes,
    flexed_bytes: bytes,
    arm_side: str | None = None,
) -> MuscleAnalysisResult:
    """Sends images to a VLM to extract specific spatial coordinates for muscles and EMS pads."""
    lax_b64 = base64.b64encode(lax_bytes).decode('utf-8')
    flexed_b64 = base64.b64encode(flexed_bytes).decode('utf-8')
    lax_media_type = _image_media_type(lax_bytes)
    flexed_media_type = _image_media_type(flexed_bytes)
    
    selected_arm = f"the subject's {arm_side} arm" if arm_side else "the same arm"
    prompt = f"""
    These are tight crops around {selected_arm}. The first image is relaxed and
    the second image is tensed/flexed. Analyze only the arm in these crops.

    1. Identify the movement by comparing the first and second images.
    2. Identify the primary visibly tensed muscles. Prioritize large arm
       movements; only consider finger movement when the arm has not changed.
    3. In the SECOND CROP, trace each affected visible muscle region with 6-8
       polygon vertices ordered clockwise around its boundary.
    4. Return all polygon and EMS pad coordinates relative to the SECOND CROP:
       - origin (0, 0) is its top-left corner
       - x increases left-to-right; y increases top-to-bottom
       - x = pixel_x / crop_width; y = pixel_y / crop_height
       - every value must be between 0.0 and 1.0
       - every vertex and pad must lie on visible arm pixels, not at a generic
         anatomical location or at the crop center by default
    5. Return at least a proximal and distal EMS pad for each muscle.
    """
    
    response = _get_client().beta.chat.completions.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{lax_media_type};base64,{lax_b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{flexed_media_type};base64,{flexed_b64}",
                            "detail": "high",
                        },
                    },
                ]
            }
        ],
        response_format=MuscleAnalysisResult,
    )

    message = response.choices[0].message
    if message.parsed is None:
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError(f"The model declined to analyze the images: {refusal}")
        raise RuntimeError("The model returned no structured analysis.")
    return message.parsed
