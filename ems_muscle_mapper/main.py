import os
import logging
import base64
import hashlib
from collections import OrderedDict
from threading import Lock
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from services.arm_region import ArmNotFoundError, extract_arm_region
from services.vlm_analyzer import (
    APIConfigurationError,
    UnsupportedImageError,
    analyze_muscle_movement,
    refine_muscle_movement,
)
from services.image_processor import build_alt_text, draw_ems_ui
from services.image_normalizer import InvalidImageError, normalize_image_orientation
from schemas import AccuracyFeedback, MuscleAnalysisResult

logger = logging.getLogger(__name__)
app = FastAPI(title="EMS Muscle Mapper")
app.mount("/static", StaticFiles(directory="templates"), name="static")

_CACHE_MAX_PAIRS = 32
_result_cache: OrderedDict[str, dict[str, object]] = OrderedDict()
_result_cache_lock = Lock()


def _image_pair_cache_key(lax_bytes: bytes, flexed_bytes: bytes) -> str:
    """Hash the ordered normalized image pair without retaining the uploads."""
    digest = hashlib.sha256()
    for image_bytes in (lax_bytes, flexed_bytes):
        digest.update(len(image_bytes).to_bytes(8, "big"))
        digest.update(image_bytes)
    return digest.hexdigest()


def _clear_result_cache() -> None:
    """Clear cached results (primarily for tests and controlled reloads)."""
    with _result_cache_lock:
        _result_cache.clear()


def _build_result(
    processed_image: bytes,
    analysis: MuscleAnalysisResult,
    analysis_id: str,
) -> dict[str, object]:
    return {
        "image_base64": base64.b64encode(processed_image).decode("ascii"),
        "alt_text": build_alt_text(analysis),
        "analysis": analysis.model_dump(mode="json"),
        "analysis_id": analysis_id,
    }


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
    
    try:
        # 1. Apply EXIF orientation and strip metadata once. Every downstream
        # service now sees the exact same upright pixels.
        normalized_lax = normalize_image_orientation(lax_bytes)
        normalized_flexed = normalize_image_orientation(flexed_bytes)
        cache_key = _image_pair_cache_key(normalized_lax, normalized_flexed)

        # Keep lookup and generation together so simultaneous identical uploads
        # cannot produce two different results before the first is cached.
        with _result_cache_lock:
            cached_result = _result_cache.get(cache_key)
            if cached_result is not None:
                _result_cache.move_to_end(cache_key)
                return dict(cached_result)

            # 2. Use the flexed image to choose an arm, then require the same
            # anatomical side in the relaxed image.
            flexed_region = extract_arm_region(normalized_flexed)
            lax_region = extract_arm_region(
                normalized_lax, preferred_side=flexed_region.side
            )

            # 3. Ask the VLM about the compact arm crops, then map its
            # crop-relative coordinates back to the full normalized image.
            crop_analysis = analyze_muscle_movement(
                lax_region.image_bytes,
                flexed_region.image_bytes,
                arm_side=flexed_region.side,
                pose_context=flexed_region.pose_prompt_context(),
            )
            refined_crop_analysis = flexed_region.refine_crop_analysis(crop_analysis)
            analysis_result = flexed_region.map_analysis_to_source(
                refined_crop_analysis
            )

            # 4. Render onto the same oriented pixels used by YOLO and OpenAI.
            processed_image = draw_ems_ui(normalized_flexed, analysis_result)
            result = _build_result(processed_image, analysis_result, cache_key)
            _result_cache[cache_key] = result
            _result_cache.move_to_end(cache_key)
            while len(_result_cache) > _CACHE_MAX_PAIRS:
                _result_cache.popitem(last=False)
            return dict(result)
        
    except (InvalidImageError, ArmNotFoundError, UnsupportedImageError) as exc:
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


@app.post("/feedback", status_code=202)
async def record_accuracy_feedback(feedback: AccuracyFeedback):
    """Record whether the user accepted the current mapping."""
    logger.info(
        "Mapping accuracy feedback: analysis_id=%s accurate=%s",
        feedback.analysis_id[:12],
        feedback.accurate,
    )
    return {"status": "received"}


@app.post("/refine")
async def refine_images(
    lax_image: UploadFile = File(...),
    flexed_image: UploadFile = File(...),
    analysis_json: str = Form(...),
    analysis_id: str = Form(...),
    feedback: str = Form(..., min_length=3, max_length=1000),
):
    """Revise a returned mapping from the user's specific visual correction."""
    lax_bytes = await lax_image.read()
    flexed_bytes = await flexed_image.read()
    feedback = feedback.strip()

    if not lax_bytes or not flexed_bytes:
        raise HTTPException(
            status_code=400,
            detail="Both original images are required to refine the mapping.",
        )
    if len(analysis_json) > 50_000:
        raise HTTPException(status_code=400, detail="The current mapping is too large.")
    if not feedback:
        raise HTTPException(
            status_code=400,
            detail="Describe what should be corrected before refining.",
        )

    try:
        current_analysis = MuscleAnalysisResult.model_validate_json(analysis_json)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail="The current mapping could not be validated.",
        ) from exc

    try:
        normalized_lax = normalize_image_orientation(lax_bytes)
        normalized_flexed = normalize_image_orientation(flexed_bytes)
        expected_id = _image_pair_cache_key(normalized_lax, normalized_flexed)
        if analysis_id != expected_id:
            raise HTTPException(
                status_code=400,
                detail="The correction does not match the uploaded image pair.",
            )

        flexed_region = extract_arm_region(normalized_flexed)
        lax_region = extract_arm_region(
            normalized_lax, preferred_side=flexed_region.side
        )
        crop_analysis = flexed_region.map_analysis_to_crop(current_analysis)
        revised_crop_analysis = refine_muscle_movement(
            lax_region.image_bytes,
            flexed_region.image_bytes,
            crop_analysis,
            feedback,
            arm_side=flexed_region.side,
            pose_context=flexed_region.pose_prompt_context(),
        )
        revised_crop_analysis = flexed_region.refine_crop_analysis(
            revised_crop_analysis
        )
        revised_analysis = flexed_region.map_analysis_to_source(
            revised_crop_analysis
        )
        processed_image = draw_ems_ui(normalized_flexed, revised_analysis)
        result = _build_result(processed_image, revised_analysis, expected_id)

        with _result_cache_lock:
            _result_cache[expected_id] = result
            _result_cache.move_to_end(expected_id)
        logger.info(
            "Mapping refined from user feedback: analysis_id=%s",
            expected_id[:12],
        )
        return result

    except HTTPException:
        raise
    except (InvalidImageError, ArmNotFoundError, UnsupportedImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except APIConfigurationError as exc:
        logger.error("OpenAI configuration error during refinement: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=503,
            detail="The server's OpenAI API key was rejected.",
        ) from exc
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=503,
            detail="The configured OpenAI project cannot access this model.",
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail="The OpenAI rate or quota limit was reached. Retry shortly.",
        ) from exc
    except APITimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The refinement request timed out. Please try again.",
        ) from exc
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to OpenAI. Check the server network.",
        ) from exc
    except APIStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI returned an upstream HTTP {exc.status_code} error.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected refinement failure.")
        raise HTTPException(
            status_code=500,
            detail="Refinement failed unexpectedly. Check the server log.",
        ) from exc
