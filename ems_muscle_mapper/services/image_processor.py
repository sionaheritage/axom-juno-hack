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
    if img is None:
        raise ValueError("The normalized flexed image could not be decoded.")
    overlay = img.copy()
    
    h, w = img.shape[:2]
    label_scale = max(0.8, min(1.5, w / 1000.0))
    
    # 1. Draw Transparent Polygons first
    for muscle in analysis.muscles:
        color_bgr = _hex_to_bgr(muscle.color_hex)
        pts = np.array(
            [
                [
                    min(w - 1, max(0, int(pt.x * w))),
                    min(h - 1, max(0, int(pt.y * h))),
                ]
                for pt in muscle.polygon_vertices_normalized
            ],
            np.int32,
        )
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color_bgr)
        
    # Blend overlay with original image
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 2. Draw Solid UI Elements (Muscle and EMS pad labels only).
    for muscle in analysis.muscles:
        color_bgr = _hex_to_bgr(muscle.color_hex)
        
        # Muscle Labels
        if len(muscle.polygon_vertices_normalized) > 0:
            lbl_x = int(muscle.polygon_vertices_normalized[0].x * w)
            lbl_y = int(muscle.polygon_vertices_normalized[0].y * h)
            cv2.putText(img, muscle.name, (lbl_x, lbl_y - 15), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 4)
            cv2.putText(img, muscle.name, (lbl_x, lbl_y - 15), cv2.FONT_HERSHEY_SIMPLEX, label_scale, color_bgr, 2)
        
        # EMS Pad Markers
        for pad in muscle.ems_pads_normalized:
            pad_x = min(w - 1, max(0, int(pad.x * w)))
            pad_y = min(h - 1, max(0, int(pad.y * h)))
            # White inner dot, distinct colored outer ring
            cv2.circle(img, (pad_x, pad_y), 8, (255, 255, 255), -1) 
            cv2.circle(img, (pad_x, pad_y), 12, (0, 150, 255), 3)     
            
            # Pad Labels (e.g., "Proximal")
            cv2.putText(img, pad.label, (pad_x + 15, pad_y + 5), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), 4)
            cv2.putText(img, pad.label, (pad_x + 15, pad_y + 5), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 2)
            
    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()


def build_alt_text(analysis: MuscleAnalysisResult) -> str:
    """Build an accessible text description for the annotated image."""
    if not analysis.muscles:
        return f"Movement detected: {analysis.movement_detected}. No muscles were mapped."

    muscle_descriptions = []
    for muscle in analysis.muscles:
        pad_labels = ", ".join(pad.label for pad in muscle.ems_pads_normalized)
        if pad_labels:
            muscle_descriptions.append(
                f"{muscle.name}, with EMS pads marked at {pad_labels}"
            )
        else:
            muscle_descriptions.append(muscle.name)

    return (
        f"Movement detected: {analysis.movement_detected}. "
        f"Highlighted muscles: {'; '.join(muscle_descriptions)}."
    )
