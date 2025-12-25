"""
FastAPI Application

Main API server with comprehensive OpenTelemetry instrumentation.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
    MetricsSnapshot,
    BatchGenerateRequest,
    BatchGenerateResponse,
    StreamGenerateRequest,
    StreamToken,
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

# Concurrency control
inference_semaphore: Optional[asyncio.Semaphore] = None
MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "1"))

# Request tracking
active_requests_count = 0
queue_wait_times = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Handles:
    - OpenTelemetry initialization
    - Model loading
    - Graceful shutdown
    """
    global inference_engine, inference_semaphore
    
    logger.info("Starting LLM Inference API")
    
    # Initialize concurrency control
    inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
    logger.info(f"Concurrency limit set to {MAX_CONCURRENT_GENERATIONS} concurrent generations")
    
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
    global inference_engine, inference_semaphore, active_requests_count
    
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
    
    # Calculate queue depth (requests waiting for semaphore)
    queue_depth = 0
    if inference_semaphore:
        queue_depth = MAX_CONCURRENT_GENERATIONS - inference_semaphore._value
        # Subtract active requests to get actual queue depth
        queue_depth = max(0, queue_depth - active_requests_count)
    
    # Update queue depth metric
    metrics.update_queue_depth(queue_depth)
    
    return HealthResponse(
        status=status,
        model_loaded=inference_engine.is_loaded(),
        model_info=model_info,
        gpu_available=torch.cuda.is_available(),
        active_requests=active_requests_count,
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
    global inference_engine, inference_semaphore
    
    if not inference_engine:
        logger.error("Inference engine not initialized")
        metrics.record_error("engine_not_initialized")
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized",
        )
    
    # Check concurrency limit
    if inference_semaphore is None:
        logger.error("Concurrency semaphore not initialized")
        metrics.record_error("semaphore_not_initialized")
        raise HTTPException(
            status_code=503,
            detail="Service not ready",
        )
    
    # Try to acquire semaphore with timeout
    try:
        acquired = await asyncio.wait_for(
            inference_semaphore.acquire(), 
            timeout=5.0  # 5 second timeout for queue
        )
        if not acquired:
            metrics.record_error("overloaded")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable - too many concurrent requests",
            )
    except asyncio.TimeoutError:
        metrics.record_error("overloaded")
        raise HTTPException(
            status_code=429,
            detail="Service overloaded - please try again later",
        )
    
    try:
        # Increment active requests counter
        active_requests_count += 1
        
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
    finally:
        # Always release the semaphore
        if inference_semaphore:
            inference_semaphore.release()
        # Always decrement active requests counter
        active_requests_count -= 1


@app.post("/generate_stream", tags=["Generation"])
async def generate_text_stream(request: GenerateRequest):
    """
    Generate text with streaming output for real-time TTFT/TPOT measurement.
    
    Returns Server-Sent Events (SSE) stream with tokens as they are generated.
    Provides accurate Time to First Token (TTFT) and Time Per Output Token (TPOT).
    
    Example usage:
    curl -X POST http://localhost:8000/generate_stream \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Explain AI", "max_tokens": 50}' \
         --no-buffer
    """
    global inference_engine, inference_semaphore
    
    if not inference_engine:
        logger.error("Inference engine not initialized")
        metrics.record_error("engine_not_initialized")
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized",
        )
    
    # Check concurrency limit
    if inference_semaphore is None:
        logger.error("Concurrency semaphore not initialized")
        metrics.record_error("semaphore_not_initialized")
        raise HTTPException(
            status_code=503,
            detail="Service not ready",
        )
    
    # Try to acquire semaphore with timeout
    try:
        acquired = await asyncio.wait_for(
            inference_semaphore.acquire(), 
            timeout=5.0
        )
        if not acquired:
            metrics.record_error("overloaded")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable - too many concurrent requests",
            )
    except asyncio.TimeoutError:
        metrics.record_error("overloaded")
        raise HTTPException(
            status_code=429,
            detail="Service overloaded - please try again later",
        )
    
    try:
        # Increment active requests counter
        active_requests_count += 1
        
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
        
        async def token_generator():
            """Async generator for streaming tokens."""
            try:
                for chunk in inference_engine.generate_stream(
                    prompt=request.prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                ):
                    if chunk.get("type") == "completion":
                        # Final completion data
                        data = f"data: {json.dumps(chunk)}\n\n"
                        yield data
                        break
                    else:
                        # Individual token
                        data = f"data: {json.dumps(chunk)}\n\n"
                        yield data
                        
            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")
                error_data = {
                    "type": "error",
                    "error": str(e),
                    "error_type": "generation_failed"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            token_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8",
            }
        )
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error("Out of memory during streaming generation")
        metrics.record_error("oom")
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=507,
            detail="Insufficient GPU memory. Try reducing max_tokens or enabling 8-bit quantization.",
        )
        
    except Exception as e:
        logger.error(f"Streaming generation failed: {e}", exc_info=True)
        metrics.record_error("generation_failed")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )
    finally:
        # Always release the semaphore
        if inference_semaphore:
            inference_semaphore.release()
        # Always decrement active requests counter
        active_requests_count -= 1


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus exposition format.
    """
    metrics_data, content_type = MetricsCollector.get_metrics_response()
    return Response(content=metrics_data, media_type=content_type)


@app.post("/generate_batch", response_model=BatchGenerateResponse, tags=["Generation"])
async def generate_text_batch(request: BatchGenerateRequest):
    """
    Generate text for multiple prompts in batch for improved throughput.
    
    Processes multiple prompts together for better GPU utilization and throughput.
    Returns individual results for each prompt along with batch-level metrics.
    
    Args:
        request: Batch generation request with multiple prompts
    
    Returns:
        BatchGenerateResponse: Individual results and batch metrics
    
    Raises:
        HTTPException: If model not loaded, service overloaded, or generation fails
    """
    global inference_engine, inference_semaphore, active_requests_count
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    # Check batch size limit
    if len(request.prompts) > 32:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large (max 32 prompts)"
        )
    
    # Acquire semaphore for batch (count as one request)
    if inference_semaphore:
        try:
            await asyncio.wait_for(inference_semaphore.acquire(), timeout=5.0)
        except asyncio.TimeoutError:
            metrics.record_error("overloaded")
            raise HTTPException(
                status_code=429,
                detail="Service overloaded - please try again later",
            )
    
    try:
        # Increment active requests counter
        active_requests_count += 1
        
        # Lazy load model if not loaded
        if not inference_engine.is_loaded():
            try:
                logger.info("Loading model on first batch request")
                inference_engine.load_model()
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                metrics.record_error("model_load_failed")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load model: {str(e)}",
                )
        
        # Generate in batch
        start_time = time.time()
        results = inference_engine.generate_batch(
            prompts=request.prompts,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
        )
        total_time = time.time() - start_time
        
        # Convert to response format
        generate_responses = [GenerateResponse(**result) for result in results]
        
        # Calculate batch metrics
        total_tokens = sum(r.total_tokens for r in generate_responses)
        throughput = total_tokens / total_time
        
        batch_response = BatchGenerateResponse(
            results=generate_responses,
            batch_size=len(request.prompts),
            total_tokens=total_tokens,
            total_time=total_time,
            throughput_tokens_per_sec=throughput,
        )
        
        logger.info(
            f"Batch generation completed - Size: {len(request.prompts)}, "
            f"Tokens: {total_tokens}, "
            f"Throughput: {throughput:.1f} tokens/sec"
        )
        
        return batch_response
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error("Out of memory error during batch generation")
        metrics.record_error("oom")
        raise HTTPException(
            status_code=507,
            detail="Insufficient GPU memory for batch. Try smaller batch size or enable 8-bit quantization.",
        )
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}", exc_info=True)
        metrics.record_error("generation_failed")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )
    finally:
        # Always release the semaphore
        if inference_semaphore:
            inference_semaphore.release()
        # Always decrement active requests counter
        active_requests_count -= 1


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


@app.post("/generate_stream", response_class=StreamingResponse, tags=["Generation"])
async def generate_text_stream(request: StreamGenerateRequest):
    """
    Generate text with streaming output.
    
    Provides real-time token-by-token generation with comprehensive metrics.
    Returns Server-Sent Events (SSE) format for easy client consumption.
    """
    global inference_engine, inference_semaphore
    
    if not inference_engine:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized",
        )
    
    if not inference_engine.is_loaded():
        inference_engine.load_model()
    
    async def stream_generator():
        """Async generator for streaming responses"""
        try:
            # Acquire semaphore for concurrency control
            if inference_semaphore:
                await inference_semaphore.acquire()
            
            # Generate with streaming
            for token_data in inference_engine.generate_stream(
                prompt=request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
            ):
                # Convert to SSE format
                yield f"data: {json.dumps(token_data)}\n\n"
                
                # Break if finished
                if token_data.get("finished", False):
                    break
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            error_data = {
                "error": True,
                "message": str(e),
                "finished": True
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            
        finally:
            # Release semaphore
            if inference_semaphore:
                inference_semaphore.release()
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
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
