import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response

from services.arm_validator import verify_arm_presence
from services.vlm_analyzer import analyze_muscle_movement
from services.image_processor import draw_ems_ui

app = FastAPI(title="EMS Muscle Mapper")

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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))