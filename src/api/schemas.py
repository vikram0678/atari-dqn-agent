"""
Pydantic schemas for API request and response validation.
"""

from pydantic import BaseModel, field_validator
from typing import List


class PredictRequest(BaseModel):
    """
    Expects a (4, 84, 84) game state as nested list.
    state[0..3] → 4 stacked grayscale frames
    Each frame is 84x84 floats in [0.0, 1.0]
    """
    state: List[List[List[float]]]

    @field_validator("state")
    @classmethod
    def validate_state(cls, v):
        if len(v) != 4:
            raise ValueError(f"Expected 4 frames, got {len(v)}")
        for i, frame in enumerate(v):
            if len(frame) != 84:
                raise ValueError(f"Frame {i} must have 84 rows, got {len(frame)}")
            for j, row in enumerate(frame):
                if len(row) != 84:
                    raise ValueError(
                        f"Frame {i}, row {j} must have 84 cols, got {len(row)}"
                    )
        return v


class PredictResponse(BaseModel):
    action: int
    q_values: List[float] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    game: str
