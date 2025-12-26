from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class HealthCheck(BaseModel):
    status: str = "Healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    service: str = "liver-segmentation-api"
    version: str = "1.0.0"

class ErrorResponse(BaseModel):
    message: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

