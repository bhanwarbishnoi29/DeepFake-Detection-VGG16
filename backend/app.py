"""
FastAPI backend for DeepFake Video Detection.
Provides REST API endpoints for health status, model diagnostics,
and video upload inference with automated temporary file cleanup.
"""

import os
# Configure sandboxed cache directories before importing ML modules
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.utils.config import load_config, get_model_save_path
from src.utils.logging_utils import setup_logger
from src.inference.predict import predict_video
from src.models.model_utils import load_trained_model

logger = setup_logger("FastAPIBackend")

# Global model state
app_state: Dict[str, Any] = {
    "model": None,
    "config": None,
    "model_path": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads model and configuration at startup."""
    try:
        cfg = load_config()
        app_state["config"] = cfg
        model_path = get_model_save_path(cfg)
        app_state["model_path"] = model_path

        if os.path.exists(model_path):
            logger.info(f"Loading trained DeepFake detection model from {model_path}...")
            app_state["model"] = load_trained_model(model_path)
            logger.info("Model loaded successfully into backend memory.")
        else:
            logger.warning(
                f"Trained model not found at {model_path}. "
                "Backend will serve /health, but /predict will require model training."
            )
    except Exception as e:
        logger.error(f"Error during backend startup initialization: {e}")

    yield

    # Teardown
    app_state["model"] = None
    logger.info("Backend service shutting down.")


app = FastAPI(
    title="Celeb-DF DeepFake Detection API",
    description="Video DeepFake Detection System using VGG16 Transfer Learning.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health Check")
async def health_check() -> Dict[str, Any]:
    """
    Returns API health status and whether model is loaded.
    """
    is_model_ready = app_state.get("model") is not None
    return {
        "status": "healthy",
        "model_loaded": is_model_ready,
        "model_path": app_state.get("model_path"),
    }


@app.post("/predict", summary="Predict DeepFake probability for uploaded video")
async def predict_endpoint(
    video: UploadFile = File(..., description="MP4, AVI, MOV, or MKV video file")
) -> Dict[str, Any]:
    """
    Receives an uploaded video file, extracts faces, runs VGG16 model inference,
    and returns detection results and confidence percentage.
    """
    if app_state.get("model") is None:
        # Check if model exists on disk and try lazy loading
        cfg = app_state.get("config") or load_config()
        model_path = get_model_save_path(cfg)
        if os.path.exists(model_path):
            try:
                app_state["model"] = load_trained_model(model_path)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load trained model: {str(e)}",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained model is not yet available. Please execute training pipeline first.",
            )

    # Validate filename extension
    filename = video.filename or ""
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {sorted(list(allowed_extensions))}",
        )

    # Save to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=ext, dir=temp_dir)
    os.close(temp_video_fd)

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Run inference
        result = predict_video(
            video_path=temp_video_path,
            model=app_state["model"],
            config=app_state.get("config"),
        )

        if result.get("prediction") == "ERROR":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result.get("error", "Error processing video file."),
            )

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "frames_analyzed": result["frames_analyzed"],
            "valid_frames": result["valid_frames"],
            "average_probability": result["average_probability"],
            "frame_probabilities": result.get("frame_probabilities", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(e)}",
        )
    finally:
        # Guarantee cleanup of temporary uploaded video file
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_video_path}: {e}")


# Optional frontend static files mount
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir) and os.path.exists(os.path.join(frontend_dir, "index.html")):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
