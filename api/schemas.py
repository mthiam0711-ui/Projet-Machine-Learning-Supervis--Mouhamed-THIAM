from typing import Optional

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: str = Field(..., description="'Yes' ou 'No'", examples=["No"])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[1])
    PhoneService: str = Field(..., examples=["No"])
    MultipleLines: str = Field(..., examples=["No phone service"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[29.85])
    TotalCharges: Optional[float] = Field(None, ge=0, examples=[29.85])


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    threshold_used: float
    risk_label: str


class HealthResponse(BaseModel):
    status: str
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    expected_features: list[str]
    decision_threshold: float
    cv_f1_mean: float
    best_params: dict
