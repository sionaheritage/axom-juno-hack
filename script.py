import os
import base64
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI

# ==============================================================================
# 1. SCHEMAS (Data Models)
# Define the strict JSON structure the VLM must return. 
# Using Pydantic guarantees the VLM output maps perfectly to our Python objects.
# ==============================================================================

class Point(BaseModel):
    x: float = Field(description="Normalized X coordinate (0.0 to 1.0)")
    y: float = Field(description="Normalized Y coordinate (0.0 to 1.0)")

class EMSPad(BaseModel):
    label: str = Field(description="e.g., 'Proximal Pad', 'Distal Pad'")
    point: Point
    
class Muscle(BaseModel):
    name: str = Field(description="Scientific name of the flexed muscle")
    polygon_vertices_normalized: List[Point] = Field(
        description="4 to 8 vertices bounding the muscle belly"
    )
    color_hex: str = Field(description="Hex color code for UI rendering")
    ems_pads_normalized: List[EMSPad] = Field(description="Where to place EMS pads")

class MuscleAnalysisResult(BaseModel):
    movement_detected: str = Field(description="e.g., 'Elbow Flexion'")
    muscles: List[Muscle]

# ==============================================================================
# 2. VALIDATION SERVICE (MediaPipe)
# Acts as a lightweight gatekeeper to reject bad photos before calling the paid API.
# ==============================================================================

class ArmValidator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=0.5
        )
        
    def validate(self, image_bytes: bytes) -> bool:
        """Verifies if an arm (shoulder, elbow, wrist) is clearly visible."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(img_rgb)
        if not results.pose_landmarks:
            return False
            
        landmarks = results.pose_landmarks.landmark
        
        # Landmark indices: Left arm (11, 13, 15), Right arm (12, 14, 16)
        left_visible = all(landmarks[i].visibility > 0.6 for i in [11, 13, 15])
        right_visible = all(landmarks[i].visibility > 0.6 for i in [12, 14, 16])
        
        return left_visible or right_visible

# ==============================================================================
# 3. VLM SERVICE (OpenAI Spatial Grounding)
# Prompts the VLM to compare the images and extract the structured spatial data.
# ==============================================================================

class VLMAnalyzer:
    def __init__(self):
        # Assumes OPENAI_API_KEY is set in your environment variables
        self.client = OpenAI() 

    def analyze_images(self, lax_bytes: bytes, flexed_bytes: bytes) -> MuscleAnalysisResult:
        lax_b64 = base64.b64encode(lax_bytes).decode("utf-8")
        flexed_b64 = base64.b64encode(flexed_bytes).decode("utf-8")
        
        prompt = """
        You are an expert biomechanics AI. Analyze these two images of an arm: 
        Image 1 is relaxed. Image 2 is flexed.
        1. Determine the movement being performed.
        2. Identify the primary muscles activated.
        3. Provide bounding polygon vertices (as normalized coordinates 0.0-1.0) for the tensed muscle in Image 2.
        4. Calculate the optimal EMS pad placement points (normalized coordinates) on the muscle belly.
        """
        
        # client.beta.chat.completions.parse strictly enforces the Pydantic schema
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{lax_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{flexed_b64}"}}
                    ]
                }
            ],
            response_format=MuscleAnalysisResult,
        )
        
        return response.choices[0].message.parsed

# ==============================================================================
# 4. IMAGE PROCESSING SERVICE (OpenCV)
# Transforms the JSON data into a beautiful UI image overlay.
# ==============================================================================

class OverlayDrawer:
    @staticmethod
    def hex_to_bgr(hex_color: str):
        """Converts web hex colors to OpenCV BGR format."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])

    @staticmethod
    def draw(image_bytes: bytes, analysis: MuscleAnalysisResult) -> bytes:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        
        overlay = img.copy()
        
        for muscle in analysis.muscles:
            color = OverlayDrawer.hex_to_bgr(muscle.color_hex)
            
            # A. Draw Semi-Transparent Polygon
            pts = np.array([[int(pt.x * w), int(pt.y * h)] for pt in muscle.polygon_vertices_normalized], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            
            # Blend the polygon into the original image
            alpha = 0.4
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            
            # B. Draw EMS Pads (Solid)
            for pad in muscle.ems_pads_normalized:
                px, py = int(pad.point.x * w), int(pad.point.y * h)
                
                # Draw high-contrast target dot
                cv2.circle(img, (px, py), 12, (255, 255, 255), -1) # White border
                cv2.circle(img, (px, py), 8, (0, 0, 255), -1)      # Red center
                
                # Label the pad with a drop shadow for legibility
                cv2.putText(img, pad.label, (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(img, pad.label, (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # C. Label the Muscle at polygon center
            cx = int(np.mean([p[0][0] for p in pts]))
            cy = int(np.mean([p[0][1] for p in pts]))
            cv2.putText(img, muscle.name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(img, muscle.name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        # Draw overarching movement text
        cv2.putText(img, f"Detected: {analysis.movement_detected}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(img, f"Detected: {analysis.movement_detected}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()

# ==============================================================================
# 5. FASTAPI ROUTER / APP (The Interface)
# ==============================================================================

app = FastAPI(title="EMS Muscle Placement System")

# Instantiate services once on startup
arm_validator = ArmValidator()
vlm_analyzer = VLMAnalyzer()
drawer = OverlayDrawer()

@app.get("/")
def get_prototype_ui():
    """Provides a very basic HTML frontend to test the upload."""
    html = """
    <html>
        <head>
            <title>EMS Onboarding</title>
            <style>body {font-family: sans-serif; padding: 2rem;}</style>
        </head>
        <body>
            <h2>EMS Pad Placement Analyzer</h2>
            <form action="/analyze" enctype="multipart/form-data" method="post">
                <label>1. Relaxed Arm Image:</label><br>
                <input name="lax_image" type="file" accept="image/*"><br><br>
                <label>2. Flexed Arm Image:</label><br>
                <input name="flexed_image" type="file" accept="image/*"><br><br>
                <input type="submit" value="Calculate Pad Placement">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/analyze")
async def analyze_endpoint(
    lax_image: UploadFile = File(...),
    flexed_image: UploadFile = File(...)
):
    lax_bytes = await lax_image.read()
    flexed_bytes = await flexed_image.read()
    
    # Step 1: Pre-validation (Fails fast and cheap)
    if not arm_validator.validate(flexed_bytes):
        raise HTTPException(
            status_code=400, 
            detail="Arm not clearly visible. Please ensure shoulder, elbow, and wrist are in frame."
        )
        
    # Step 2: VLM JSON Extraction
    try:
        analysis_result = vlm_analyzer.analyze_images(lax_bytes, flexed_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM Analysis failed: {str(e)}")
        
    # Step 3: Draw overlays on the flexed image
    annotated_image_bytes = drawer.draw(flexed_bytes, analysis_result)
    
    # Step 4: Return the annotated image directly to the browser/UI
    return Response(content=annotated_image_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    # Run server locally: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)