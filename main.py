"""
Airfare Price Prediction API (Optimized for 512MB RAM Deployments)
=================================================================
A lightweight, memory-efficient FastAPI microservice that serves the trained
Scikit-Learn Airfare Price Prediction Pipeline.

Optimizations for Render Free Tier (512 MiB limit):
  - Model loaded strictly ONCE at startup via lifespan handler.
  - Regressor configured with n_jobs=1 to avoid multi-thread/process memory inflation.
  - Explicit garbage collection immediately after model deserialization.
  - Multi-path resolution for model artifacts (supports root and 'ml/' directory).
  - Production-ready port binding for Render ($PORT and 0.0.0.0).
  - Single-worker architecture preventing duplicate process memory.
"""

import os
import gc
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

# Candidates for model and metrics paths (supports both root and ml/ subfolder)
def resolve_file_path(default_name: str, alt_subfolder: str = "ml") -> str:
    candidates = [
        os.environ.get(default_name.upper().replace(".", "_")),
        os.path.join(alt_subfolder, default_name),
        default_name,
        os.path.join(os.path.dirname(__file__), alt_subfolder, default_name),
        os.path.join(os.path.dirname(__file__), default_name),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return default_name

MODEL_PATH = resolve_file_path("airfare_model.pkl")
METRICS_PATH = resolve_file_path("model_metrics.json")

# Deterministic route distance lookup table (km) between Indian metro airports
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

# Global in-memory singleton
model_pipeline = None
model_metadata = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the machine learning pipeline and metadata ONCE on startup."""
    global model_pipeline, model_metadata
    logger.info("Initializing Airfare Prediction API...")
    
    # Resolve and load model pipeline
    active_model_path = resolve_file_path("airfare_model.pkl")
    if os.path.exists(active_model_path):
        try:
            logger.info("Loading model pipeline from %s ...", active_model_path)
            model_pipeline = joblib.load(active_model_path)
            
            # Ensure single-thread inference to prevent joblib threadpool memory overhead
            if hasattr(model_pipeline, "named_steps") and "regressor" in model_pipeline.named_steps:
                reg = model_pipeline.named_steps["regressor"]
                if hasattr(reg, "n_jobs"):
                    reg.n_jobs = 1
            
            # Immediately reclaim deserialization memory buffers
            gc.collect()
            logger.info("Model pipeline loaded successfully (single worker mode enabled).")
        except Exception as e:
            logger.error("Failed to load model pipeline from %s: %s", active_model_path, e)
            model_pipeline = None
    else:
        logger.warning("Model file not found at %s.", active_model_path)

    # Load metadata
    active_metrics_path = resolve_file_path("model_metrics.json")
    if os.path.exists(active_metrics_path):
        try:
            with open(active_metrics_path, "r") as f:
                model_metadata = json.load(f)
            logger.info("Loaded model metrics from %s", active_metrics_path)
        except Exception as e:
            logger.warning("Could not read %s: %s", active_metrics_path, e)
            model_metadata = {}
    
    yield
    logger.info("Shutting down Airfare Prediction API...")


app = FastAPI(
    title="Airfare Price Prediction API",
    description=(
        "Production-ready Machine Learning REST API for predicting domestic airline ticket prices in India. "
        "Engineered for high performance within a strict 512 MB RAM footprint."
    ),
    version="1.1.0",
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
    rev_key = (destination.strip(), source.strip())
    if rev_key in ROUTE_DISTANCE_MAP:
        return ROUTE_DISTANCE_MAP[rev_key]
    return 1168.23


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    """Root confirmation endpoint."""
    return {
        "message": "Airfare Price Prediction API is running successfully.",
        "version": "1.1.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint verifying model loading and system readiness."""
    is_loaded = model_pipeline is not None
    model_name = model_metadata.get("selected_model", "RandomForestRegressor (Tuned Pipeline)")
    eval_metrics = model_metadata.get("evaluation_metrics", {})
    
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_name": model_name,
        "performance_metrics": {
            "test_r2": eval_metrics.get("test_r2", 0.9856),
            "test_mae_inr": eval_metrics.get("test_mae", 1172.17),
            "test_rmse_inr": eval_metrics.get("test_rmse", 2727.86)
        }
    }


@app.post("/predict", response_model=FlightPredictionResponse, tags=["Prediction"])
def predict_airfare(request: FlightPredictionRequest):
    """
    Predict flight airfare for a single flight request.
    Validates features, applies fitted encoders and scalers,
    and returns estimated fare in INR.
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained model pipeline not loaded. Please verify model file exists."
        )

    try:
        resolved_distance = resolve_route_distance(
            request.source_city,
            request.destination_city,
            request.route_distance
        )

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

        # Predict using single row DataFrame
        input_df = pd.DataFrame([feature_dict])
        predicted_price = float(model_pipeline.predict(input_df)[0])
        predicted_price = max(1000.0, predicted_price)

        model_name = model_metadata.get("selected_model", "RandomForestRegressor (Tuned Pipeline)")

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
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained model pipeline not loaded."
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
        model_name = model_metadata.get("selected_model", "RandomForestRegressor (Tuned Pipeline)")

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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting server on host 0.0.0.0, port %d (workers=1)", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
