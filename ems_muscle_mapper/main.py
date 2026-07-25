from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from services.arm_validator import verify_arm_presence
from services.vlm_analyzer import analyze_muscle_movement
from services.image_processor import draw_ems_ui

app = FastAPI(title="EMS Muscle Mapper")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renders the frontend upload interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def process_images(lax_image: UploadFile = File(...), flexed_image: UploadFile = File(...)):
    """Receives the two images, processes them, and returns an annotated image."""
    lax_bytes = await lax_image.read()
    flexed_bytes = await flexed_image.read()
    
    # 1. Edge-Compute Validation (MediaPipe)
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))