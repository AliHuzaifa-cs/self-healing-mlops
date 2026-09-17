"""
Request/response validation schemas for the prediction API.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    return_lag1: float = Field(..., description="1-day lagged daily return")
    return_lag2: float = Field(..., description="2-day lagged daily return")
    return_lag3: float = Field(..., description="3-day lagged daily return")
    price_vs_ma5: float = Field(..., description="(close - MA5) / MA5")
    price_vs_ma20: float = Field(..., description="(close - MA20) / MA20")
    volatility_5: float = Field(..., description="5-day rolling std of returns")
    volatility_10: float = Field(..., description="10-day rolling std of returns")
    volume_change: float = Field(..., description="% change in volume vs previous day")
    volume_vs_ma5: float = Field(..., description="(volume - volMA5) / volMA5")
    hl_range: float = Field(..., description="(high - low) / close")


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool