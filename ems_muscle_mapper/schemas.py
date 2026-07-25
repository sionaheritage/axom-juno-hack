from pydantic import BaseModel
from typing import List

class Coordinate(BaseModel):
    x: float
    y: float

class EMSPad(BaseModel):
    label: str
    x: float
    y: float

class MuscleGroup(BaseModel):
    name: str
    polygon_vertices_normalized: List[Coordinate]
    color_hex: str
    ems_pads_normalized: List[EMSPad]

class MuscleAnalysisResult(BaseModel):
    movement_detected: str
    muscles: List[MuscleGroup]