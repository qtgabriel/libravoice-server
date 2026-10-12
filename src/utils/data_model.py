from pydantic import BaseModel
from typing import List, Dict, Any, Union

class LandmarkPayload(BaseModel):
    frames: Union[List[List[Dict[str, float]]], List[Dict[str, float]], None] = None
    landmarks: Union[List[List[Dict[str, float]]], List[Dict[str, float]], None] = None

class TtlRequest(BaseModel):
    text: str 