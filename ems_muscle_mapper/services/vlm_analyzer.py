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

def analyze_muscle_movement(lax_bytes: bytes, flexed_bytes: bytes) -> MuscleAnalysisResult:
    """Sends images to a VLM to extract specific spatial coordinates for muscles and EMS pads."""
    lax_b64 = base64.b64encode(lax_bytes).decode('utf-8')
    flexed_b64 = base64.b64encode(flexed_bytes).decode('utf-8')
    lax_media_type = _image_media_type(lax_bytes)
    flexed_media_type = _image_media_type(flexed_bytes)
    
    prompt = """
    Analyze these two images: the first is a relaxed arm, the second is a tensed/flexed arm.
    1. Identify the movement being performed.
    2. Identify the primary tensed muscles.
    3. Provide the bounding polygon vertices (as normalized coordinates 0.0 to 1.0) for each tensed muscle in the FLEXED image. Keep polygons simple (4-6 points).
    4. Calculate the optimal EMS pad placement points (normalized coordinates 0.0 to 1.0) for these muscles. A muscle typically needs a Proximal and Distal pad.
    """
    
    response = _get_client().beta.chat.completions.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{lax_media_type};base64,{lax_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:{flexed_media_type};base64,{flexed_b64}"}}
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
