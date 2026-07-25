import os
import base64
from openai import OpenAI
from schemas import MuscleAnalysisResult

# 3b. The VLM API Integrator
# Jack's API key (thanks Jack)
client = OpenAI(api_key="1BVflTUJyl3kdkDkEjJ19TnQNS5N0pwVu7gIXgDo6EGPpbQrvIHaEbVFuEROh5oc")

# Alt: get OPENAI_API_KEY environment variable set
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_muscle_movement(lax_bytes: bytes, flexed_bytes: bytes) -> MuscleAnalysisResult:
    """Sends images to a VLM to extract specific spatial coordinates for muscles and EMS pads."""
    lax_b64 = base64.b64encode(lax_bytes).decode('utf-8')
    flexed_b64 = base64.b64encode(flexed_bytes).decode('utf-8')
    
    prompt = """
    Analyze these two images: the first is a relaxed arm, the second is a tensed/flexed arm.
    1. Identify the movement being performed.
    2. Identify the primary tensed muscles.
    3. Provide the bounding polygon vertices (as normalized coordinates 0.0 to 1.0) for each tensed muscle in the FLEXED image. Keep polygons simple (4-6 points).
    4. Calculate the optimal EMS pad placement points (normalized coordinates 0.0 to 1.0) for these muscles. A muscle typically needs a Proximal and Distal pad.
    """
    
    response = client.beta.chat.completions.parse(
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