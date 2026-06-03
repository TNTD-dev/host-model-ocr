from pydantic import BaseModel


class OCRResponse(BaseModel):
    text: str
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    kaggle_reachable: bool
