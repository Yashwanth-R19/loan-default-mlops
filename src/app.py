"""FastAPI prediction service for loan default risk."""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_model, predict_one
from src.utils import PROJECT_ROOT

STATIC_DIR = PROJECT_ROOT / "src" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # load once at startup instead of on the first request
    yield


app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts the probability that a loan applicant will default, "
    "using an XGBoost model tuned and tracked with MLflow/DagsHub.",
    version="1.0.0",
    lifespan=lifespan,
)


class LoanApplication(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    Income: int = Field(..., ge=0, description="Annual income in dollars")
    LoanAmount: int = Field(..., ge=0, description="Requested loan amount in dollars")
    CreditScore: int = Field(..., ge=300, le=850, description="Credit score (FICO range)")
    MonthsEmployed: int = Field(..., ge=0, description="Months at current employer")
    NumCreditLines: int = Field(..., ge=0, description="Number of open credit lines")
    InterestRate: float = Field(..., ge=0, le=50, description="Loan interest rate (%)")
    LoanTerm: int = Field(..., ge=1, description="Loan term in months")
    DTIRatio: float = Field(..., ge=0, le=1, description="Debt-to-income ratio")
    Education: Literal["High School", "Bachelor's", "Master's", "PhD"]
    EmploymentType: Literal["Full-time", "Part-time", "Self-employed", "Unemployed"]
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    HasMortgage: Literal["Yes", "No"]
    HasDependents: Literal["Yes", "No"]
    LoanPurpose: Literal["Auto", "Business", "Education", "Home", "Other"]
    HasCoSigner: Literal["Yes", "No"]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Age": 35,
                "Income": 65000,
                "LoanAmount": 15000,
                "CreditScore": 680,
                "MonthsEmployed": 48,
                "NumCreditLines": 3,
                "InterestRate": 12.5,
                "LoanTerm": 36,
                "DTIRatio": 0.35,
                "Education": "Bachelor's",
                "EmploymentType": "Full-time",
                "MaritalStatus": "Married",
                "HasMortgage": "Yes",
                "HasDependents": "No",
                "LoanPurpose": "Auto",
                "HasCoSigner": "No",
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = predicted default, 0 = predicted no default")
    default_probability: float = Field(..., description="Model's predicted probability of default")
    risk_label: str = Field(..., description="Low / Medium / High risk bucket")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication) -> dict:
    return predict_one(application.model_dump())
