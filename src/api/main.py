"""
FastAPI Application

Main API server with comprehensive OpenTelemetry instrumentation.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ErrorResponse,
)
from ..inference import InferenceEngine
from ..telemetry import setup_telemetry, shutdown_telemetry, MetricsCollector


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global inference engine instance
inference_engine: Optional[InferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Handles:
    - OpenTelemetry initialization
    - Model loading
    - Graceful shutdown
    """
    global inference_engine
    
    logger.info("Starting LLM Inference API")
    
    # Initialize OpenTelemetry
    try:
        setup_telemetry()
        logger.info("OpenTelemetry initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
    
    # Initialize inference engine
    try:
        model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
        device = os.getenv("MODEL_DEVICE", "cuda")
        cache_dir = os.getenv("MODEL_CACHE_DIR", "./models")
        load_in_8bit = os.getenv("MODEL_LOAD_IN_8BIT", "true").lower() == "true"
        max_length = int(os.getenv("MODEL_MAX_LENGTH", "2048"))
        
        inference_engine = InferenceEngine(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            load_in_8bit=load_in_8bit,
            max_length=max_length,
        )
        
        logger.info("Inference engine initialized")
        
        # Load model on startup (optional - can lazy load on first request)
        # Uncomment to preload model:
        # logger.info("Preloading model...")
        # inference_engine.load_model()
        # logger.info("Model preloaded")
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Inference API")
    
    if inference_engine:
        inference_engine.unload_model()
    
    shutdown_telemetry()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="LLM Latency Bottleneck Analysis API",
    description="Production-grade LLM inference with comprehensive performance instrumentation",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Get metrics collector
metrics = MetricsCollector()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    import time
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    metrics.record_error("unhandled_exception")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An internal error occurred",
        },
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "LLM Latency Bottleneck Analysis API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service status, model loading state, and system information.
    """
    global inference_engine
    
    if not inference_engine:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_info=None,
            gpu_available=torch.cuda.is_available(),
            active_requests=0,
        )
    
    model_info = inference_engine.get_model_info() if inference_engine.is_loaded() else None
    
    # Determine status
    status = "healthy" if inference_engine.is_loaded() else "degraded"
    
    # Update system metrics
    metrics.update_system_metrics()
    
    return HealthResponse(
        status=status,
        model_loaded=inference_engine.is_loaded(),
        model_info=model_info,
        gpu_available=torch.cuda.is_available(),
        active_requests=0,  # Could track this with middleware
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_text(request: GenerateRequest):
    """
    Generate text from a prompt.
    
    Tracks comprehensive performance metrics including:
    - Time to First Token (TTFT)
    - Time Per Output Token (TPOT)
    - Memory usage
    - Token counts
    
    Returns generated text with detailed performance metrics.
    """
    global inference_engine
    
    if not inference_engine:
        logger.error("Inference engine not initialized")
        metrics.record_error("engine_not_initialized")
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized",
        )
    
    # Lazy load model if not loaded
    if not inference_engine.is_loaded():
        try:
            logger.info("Loading model on first request")
            inference_engine.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            metrics.record_error("model_load_failed")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load model: {str(e)}",
            )
    
    # Generate text
    try:
        result = inference_engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
        )
        
        return GenerateResponse(**result)
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error("Out of memory during generation")
        metrics.record_error("oom")
        
        # Try to clear cache
        torch.cuda.empty_cache()
        
        raise HTTPException(
            status_code=507,
            detail="Insufficient GPU memory. Try reducing max_tokens or enabling 8-bit quantization.",
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        metrics.record_error("generation_failed")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus exposition format.
    """
    metrics_data, content_type = MetricsCollector.get_metrics_response()
    return Response(content=metrics_data, media_type=content_type)


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Get model information and status.
    
    Returns detailed information about the loaded model.
    """
    global inference_engine
    
    if not inference_engine:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized",
        )
    
    return inference_engine.get_model_info()


@app.post("/model/load", tags=["Model"])
async def load_model():
    """
    Manually trigger model loading.
    
    Useful for preloading the model before handling requests.
    """
    global inference_engine
    
    if not inference_engine:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized",
        )
    
    if inference_engine.is_loaded():
        return {"message": "Model already loaded", "status": "success"}
    
    try:
        inference_engine.load_model()
        return {"message": "Model loaded successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}",
        )


@app.post("/model/unload", tags=["Model"])
async def unload_model():
    """
    Unload model from memory.
    
    Useful for freeing GPU memory.
    """
    global inference_engine
    
    if not inference_engine:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized",
        )
    
    if not inference_engine.is_loaded():
        return {"message": "Model not loaded", "status": "success"}
    
    try:
        inference_engine.unload_model()
        return {"message": "Model unloaded successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
