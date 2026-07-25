import cv2
import numpy as np
from schemas import MuscleAnalysisResult

# 3c. OpenCV Render Engine

def _hex_to_bgr(hex_color: str):
    """Converts standard HEX code to OpenCV's BGR color space."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

def draw_ems_ui(image_bytes: bytes, analysis: MuscleAnalysisResult) -> bytes:
    """Overlays polygons, dots, and labels over the flexed image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    overlay = img.copy()
    
    h, w = img.shape[:2]
    
    # 1. Draw Transparent Polygons first
    for muscle in analysis.muscles:
        color_bgr = _hex_to_bgr(muscle.color_hex)
        pts = np.array([[int(pt.x * w), int(pt.y * h)] for pt in muscle.polygon_vertices_normalized], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color_bgr)
        
    # Blend overlay with original image
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 2. Draw Solid UI Elements (Text & EMS Pads)
    cv2.putText(img, f"Movement: {analysis.movement_detected}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img, f"Movement: {analysis.movement_detected}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    for muscle in analysis.muscles:
        color_bgr = _hex_to_bgr(muscle.color_hex)
        
        # Muscle Labels
        if len(muscle.polygon_vertices_normalized) > 0:
            lbl_x = int(muscle.polygon_vertices_normalized[0].x * w)
            lbl_y = int(muscle.polygon_vertices_normalized[0].y * h)
            cv2.putText(img, muscle.name, (lbl_x, lbl_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            cv2.putText(img, muscle.name, (lbl_x, lbl_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        
        # EMS Pad Markers
        for pad in muscle.ems_pads_normalized:
            pad_x, pad_y = int(pad.x * w), int(pad.y * h)
            # White inner dot, distinct colored outer ring
            cv2.circle(img, (pad_x, pad_y), 8, (255, 255, 255), -1) 
            cv2.circle(img, (pad_x, pad_y), 12, (0, 150, 255), 3)     
            
            # Pad Labels (e.g., "Proximal")
            cv2.putText(img, pad.label, (pad_x + 15, pad_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, pad.label, (pad_x + 15, pad_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()