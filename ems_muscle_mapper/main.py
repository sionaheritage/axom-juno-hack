import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from services.arm_validator import verify_arm_presence
from services.vlm_analyzer import (
    APIConfigurationError,
    UnsupportedImageError,
    analyze_muscle_movement,
)
from services.image_processor import draw_ems_ui

logger = logging.getLogger(__name__)
app = FastAPI(title="EMS Muscle Mapper")
app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Renders the frontend upload interface natively without a templating engine."""
    html_path = os.path.join("templates", "index.html")
    
    # Read the file natively
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found in templates folder.")
        
    return HTMLResponse(content=html_content)

@app.post("/analyze")
async def process_images(lax_image: UploadFile = File(...), flexed_image: UploadFile = File(...)):
    """Receives the two images, processes them, and returns an annotated image."""
    lax_bytes = await lax_image.read()
    flexed_bytes = await flexed_image.read()

    if not lax_bytes or not flexed_bytes:
        raise HTTPException(status_code=400, detail="Both uploaded images must contain data.")
    
    # 1. Edge-Compute Validation (YOLO-Pose)
    if not verify_arm_presence(lax_bytes) or not verify_arm_presence(flexed_bytes):
        raise HTTPException(
            status_code=400, 
            detail="Could not detect a clear arm in one or both images. Ensure your elbow and wrist are visible."
        )
        
    try:
        # 2. VLM Spatial Grounding
        analysis_result = analyze_muscle_movement(lax_bytes, flexed_bytes)
        
        # 3. OpenCV Rendering
        processed_image = draw_ems_ui(flexed_bytes, analysis_result)
        
        return Response(content=processed_image, media_type="image/jpeg")
        
    except UnsupportedImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except APIConfigurationError as exc:
        logger.error("OpenAI configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        logger.warning("OpenAI rejected the configured API credentials.")
        raise HTTPException(
            status_code=503,
            detail="The server's OpenAI API key was rejected. Replace OPENAI_API_KEY and restart the server.",
        ) from exc
    except PermissionDeniedError as exc:
        logger.warning("OpenAI denied model or project access.")
        raise HTTPException(
            status_code=503,
            detail="The configured OpenAI project does not have access to this model.",
        ) from exc
    except RateLimitError as exc:
        logger.warning("OpenAI rate or quota limit reached.")
        raise HTTPException(
            status_code=429,
            detail="The OpenAI rate or quota limit was reached. Check project billing or retry shortly.",
        ) from exc
    except APITimeoutError as exc:
        logger.warning("OpenAI request timed out.")
        raise HTTPException(
            status_code=504,
            detail="The OpenAI request timed out. Please try again.",
        ) from exc
    except APIConnectionError as exc:
        logger.warning("Could not connect to OpenAI: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Could not connect to OpenAI. Check the server network and OPENAI_BASE_URL.",
        ) from exc
    except APIStatusError as exc:
        logger.warning("OpenAI returned HTTP %s.", exc.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI returned an upstream HTTP {exc.status_code} error.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected analysis failure.")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed unexpectedly. Check the server log for details.",
        ) from exc
