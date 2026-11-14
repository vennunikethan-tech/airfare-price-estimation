"""
Airfare Price Prediction API
============================
A high-performance FastAPI service that serves the trained Scikit-Learn
Airfare Price Prediction Pipeline.

Endpoints:
  - GET  /         : Root welcome endpoint
  - GET  /health   : System health & model status check
  - POST /predict  : Single flight fare prediction
  - POST /predict/batch : Batch flight fare prediction
"""

import os
import json
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("airfare_api")

# Constants & Paths
MODEL_PATH = "airfare_model.pkl"
METRICS_PATH = "model_metrics.json"

# Deterministic route distance lookup table (in kilometers) between major Indian metro airports
# Allows route_distance to be automatically derived if omitted by the user.
ROUTE_DISTANCE_MAP = {
    ("Bangalore", "Chennai"): 267.9,
    ("Bangalore", "Delhi"): 1708.8,
    ("Bangalore", "Hyderabad"): 456.1,
    ("Bangalore", "Kolkata"): 1546.8,
    ("Bangalore", "Mumbai"): 834.4,
    ("Chennai", "Bangalore"): 267.9,
    ("Chennai", "Delhi"): 1759.4,
    ("Chennai", "Hyderabad"): 507.8,
    ("Chennai", "Kolkata"): 1385.2,
    ("Chennai", "Mumbai"): 1033.5,
    ("Delhi", "Bangalore"): 1708.8,
    ("Delhi", "Chennai"): 1759.4,
    ("Delhi", "Hyderabad"): 1265.6,
    ("Delhi", "Kolkata"): 1312.4,
    ("Delhi", "Mumbai"): 1137.1,
    ("Hyderabad", "Bangalore"): 456.1,
    ("Hyderabad", "Chennai"): 507.8,
    ("Hyderabad", "Delhi"): 1265.6,
    ("Hyderabad", "Kolkata"): 1207.2,
    ("Hyderabad", "Mumbai"): 622.7,
    ("Kolkata", "Bangalore"): 1546.8,
    ("Kolkata", "Chennai"): 1385.2,
    ("Kolkata", "Delhi"): 1312.4,
    ("Kolkata", "Hyderabad"): 1207.2,
    ("Kolkata", "Mumbai"): 1665.7,
    ("Mumbai", "Bangalore"): 834.4,
    ("Mumbai", "Chennai"): 1033.5,
    ("Mumbai", "Delhi"): 1137.1,
    ("Mumbai", "Hyderabad"): 622.7,
    ("Mumbai", "Kolkata"): 1665.7,
}

# Global state holders
model_pipeline = None
model_metadata = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the machine learning pipeline and metadata on startup."""
    global model_pipeline, model_metadata
    logger.info("Initializing Airfare Prediction API...")
    
    # Load model pipeline
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            logger.info("Successfully loaded model pipeline from %s", MODEL_PATH)
        except Exception as e:
            logger.error("Failed to load model pipeline from %s: %s", MODEL_PATH, e)
            model_pipeline = None
    else:
        logger.warning("Model file %s not found on startup.", MODEL_PATH)

    # Load metadata
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                model_metadata = json.load(f)
            logger.info("Successfully loaded model metrics from %s", METRICS_PATH)
        except Exception as e:
            logger.warning("Could not read %s: %s", METRICS_PATH, e)
            model_metadata = {}
    
    yield
    logger.info("Shutting down Airfare Prediction API...")


app = FastAPI(
    title="Airfare Price Prediction API",
    description=(
        "Production-ready Machine Learning REST API for predicting domestic airline ticket prices in India. "
        "Powered by an optimized Scikit-learn Pipeline with automated categorical encoding, feature scaling, "
        "and an ensemble Random Forest regression model."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------------------

class FlightPredictionRequest(BaseModel):
    airline: str = Field(
        ...,
        description="Operating airline (e.g., 'SpiceJet', 'AirAsia', 'Vistara', 'Indigo', 'Air_India', 'GO_FIRST')",
        examples=["SpiceJet"]
    )
    source_city: str = Field(
        ...,
        description="Departure city (e.g., 'Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai')",
        examples=["Delhi"]
    )
    destination_city: str = Field(
        ...,
        description="Arrival city (e.g., 'Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai')",
        examples=["Mumbai"]
    )
    departure_time: str = Field(
        ...,
        description="Time slot of departure (e.g., 'Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night')",
        examples=["Evening"]
    )
    arrival_time: str = Field(
        ...,
        description="Time slot of arrival (e.g., 'Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night')",
        examples=["Night"]
    )
    stops: str = Field(
        ...,
        description="Number of intermediate stops ('zero', 'one', 'two_or_more')",
        examples=["zero"]
    )
    class_type: str = Field(
        ...,
        alias="class",
        description="Cabin travel class ('Economy' or 'Business')",
        examples=["Economy"]
    )
    duration: float = Field(
        ...,
        gt=0.0,
        description="Total journey duration in hours (e.g., 2.17)",
        examples=[2.17]
    )
    days_left: int = Field(
        ...,
        ge=1,
        le=365,
        description="Number of days remaining before flight departure (1 to 365)",
        examples=[1]
    )
    route_distance: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Approximate straight-line airport distance in kilometers. "
            "Optional: if omitted, it is automatically resolved from standard airport coordinates."
        ),
        examples=[1137.1]
    )

    @field_validator("stops")
    @classmethod
    def normalize_stops(cls, v: str) -> str:
        clean = v.strip().lower()
        mapping = {
            "0": "zero",
            "zero": "zero",
            "none": "zero",
            "non-stop": "zero",
            "direct": "zero",
            "1": "one",
            "one": "one",
            "2": "two_or_more",
            "two": "two_or_more",
            "two_or_more": "two_or_more",
            "2+": "two_or_more"
        }
        return mapping.get(clean, clean)

    @field_validator("class_type")
    @classmethod
    def normalize_class(cls, v: str) -> str:
        clean = v.strip().capitalize()
        if clean in ["Economy", "Business"]:
            return clean
        return clean

    class Config:
        populate_by_name = True


class FlightPredictionResponse(BaseModel):
    predicted_airfare: float = Field(..., description="Estimated ticket price rounded to 2 decimal places")
    currency: str = Field(default="INR", description="Currency code (Indian Rupee)")
    selected_model: str = Field(..., description="Underlying machine learning model name")
    prediction_status: str = Field(..., description="Status of the prediction ('success' or 'error')")
    features_used: dict = Field(..., description="Summary of features passed into the prediction pipeline")


class BatchFlightPredictionRequest(BaseModel):
    flights: List[FlightPredictionRequest]


class BatchFlightPredictionResponse(BaseModel):
    predictions: List[FlightPredictionResponse]
    total_processed: int
    currency: str = "INR"


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def resolve_route_distance(source: str, destination: str, provided_distance: Optional[float]) -> float:
    """Resolve distance from request or airport lookup table, defaulting safely."""
    if provided_distance is not None and provided_distance > 0:
        return float(provided_distance)
    key = (source.strip(), destination.strip())
    if key in ROUTE_DISTANCE_MAP:
        return ROUTE_DISTANCE_MAP[key]
    # Reverse lookup if applicable
    rev_key = (destination.strip(), source.strip())
    if rev_key in ROUTE_DISTANCE_MAP:
        return ROUTE_DISTANCE_MAP[rev_key]
    # Default fallback average metro distance
    return 1168.23


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    """Root confirmation endpoint."""
    return {
        "message": "Airfare Price Prediction API is running successfully.",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint verifying model loading and system readiness."""
    is_loaded = model_pipeline is not None
    model_name = model_metadata.get("selected_model", "RandomForestRegressor (Pipeline)")
    eval_metrics = model_metadata.get("evaluation_metrics", {})
    
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_name": model_name,
        "model_file": MODEL_PATH,
        "performance_metrics": {
            "test_r2": eval_metrics.get("test_r2"),
            "test_mae_inr": eval_metrics.get("test_mae"),
            "test_rmse_inr": eval_metrics.get("test_rmse")
        }
    }


@app.post("/predict", response_model=FlightPredictionResponse, tags=["Prediction"])
def predict_airfare(request: FlightPredictionRequest):
    """
    Predict flight airfare for a single flight request.
    
    Validates features, applies fitted encoders and scalers,
    and returns estimated fare in INR.
    """
    global model_pipeline
    if model_pipeline is None:
        if os.path.exists(MODEL_PATH):
            model_pipeline = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained model pipeline not available. Please ensure airfare_model.pkl exists."
            )

    try:
        # Resolve route distance
        resolved_distance = resolve_route_distance(
            request.source_city,
            request.destination_city,
            request.route_distance
        )

        # Assemble feature row matching exact training pipeline columns
        feature_dict = {
            "airline": request.airline,
            "source_city": request.source_city,
            "departure_time": request.departure_time,
            "stops": request.stops,
            "arrival_time": request.arrival_time,
            "destination_city": request.destination_city,
            "class": request.class_type,
            "duration": float(request.duration),
            "days_left": int(request.days_left),
            "route_distance": float(resolved_distance)
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([feature_dict])

        # Predict
        predicted_price = float(model_pipeline.predict(input_df)[0])
        
        # Guard against anomalous negative bounds
        predicted_price = max(1000.0, predicted_price)

        model_name = model_metadata.get("selected_model", "RandomForestRegressor (Pipeline)")

        return FlightPredictionResponse(
            predicted_airfare=round(predicted_price, 2),
            currency="INR",
            selected_model=model_name,
            prediction_status="success",
            features_used=feature_dict
        )

    except Exception as e:
        logger.exception("Error occurred while generating prediction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchFlightPredictionResponse, tags=["Prediction"])
def predict_airfare_batch(batch_request: BatchFlightPredictionRequest):
    """Predict flight airfare for a batch of flight itineraries."""
    global model_pipeline
    if model_pipeline is None:
        if os.path.exists(MODEL_PATH):
            model_pipeline = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained model pipeline not available."
            )

    try:
        rows = []
        for req in batch_request.flights:
            dist = resolve_route_distance(req.source_city, req.destination_city, req.route_distance)
            rows.append({
                "airline": req.airline,
                "source_city": req.source_city,
                "departure_time": req.departure_time,
                "stops": req.stops,
                "arrival_time": req.arrival_time,
                "destination_city": req.destination_city,
                "class": req.class_type,
                "duration": float(req.duration),
                "days_left": int(req.days_left),
                "route_distance": float(dist)
            })

        input_df = pd.DataFrame(rows)
        preds = model_pipeline.predict(input_df)
        model_name = model_metadata.get("selected_model", "RandomForestRegressor (Pipeline)")

        results = []
        for feat, pred in zip(rows, preds):
            results.append(FlightPredictionResponse(
                predicted_airfare=round(max(1000.0, float(pred)), 2),
                currency="INR",
                selected_model=model_name,
                prediction_status="success",
                features_used=feat
            ))

        return BatchFlightPredictionResponse(
            predictions=results,
            total_processed=len(results),
            currency="INR"
        )
    except Exception as e:
        logger.exception("Batch prediction error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )

