"""
Inference Engine

LLM inference engine with comprehensive performance instrumentation.
Loads model, manages generation, and tracks detailed timing metrics.
"""

import logging
import os
from typing import Optional, Dict, Any, Iterator
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from opentelemetry import trace

from ..telemetry import get_tracer, MetricsCollector
from ..telemetry.metrics_collector import InferenceMetrics
from .metrics import InferenceTimer, TokenTimings


logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Production-grade LLM inference engine with observability.
    
    Features:
    - Lazy model loading
    - 8-bit quantization support
    - Token-by-token generation with timing
    - Comprehensive metrics collection
    - Distributed tracing integration
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        load_in_8bit: bool = True,
        max_length: int = 2048,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (cuda/cpu)
            cache_dir: Directory for caching model files
            load_in_8bit: Enable 8-bit quantization for memory efficiency
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", "./models")
        self.load_in_8bit = load_in_8bit and self.device == "cuda"
        self.max_length = max_length
        
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        self.tracer = get_tracer(__name__)
        self.metrics = MetricsCollector()
        
        self._model_loaded = False
        
        logger.info(
            f"InferenceEngine initialized: {model_name} on {self.device}, "
            f"8-bit: {self.load_in_8bit}"
        )
    
    def load_model(self) -> None:
        """
        Load model and tokenizer with proper configuration.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._model_loaded:
            logger.info("Model already loaded")
            return
        
        with self.tracer.start_as_current_span("load_model") as span:
            span.set_attribute("model.name", self.model_name)
            span.set_attribute("model.device", self.device)
            span.set_attribute("model.quantization", self.load_in_8bit)
            
            try:
                logger.info(f"Loading model: {self.model_name}")
                
                # Create cache directory
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                
                # Load tokenizer
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )
                
                # Set padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                span.add_event("tokenizer_loaded")
                
                # Configure quantization
                quantization_config = None
                if self.load_in_8bit:
                    logger.info("Configuring 8-bit quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                
                # Load model
                logger.info("Loading model weights...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                
                # Move to device if not using device_map
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                
                span.add_event("model_loaded")
                self._model_loaded = True
                self.metrics.set_model_loaded(True)
                
                # Log memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    logger.info(
                        f"GPU Memory - Allocated: {memory_allocated:.2f}GB, "
                        f"Reserved: {memory_reserved:.2f}GB"
                    )
                    span.set_attribute("model.gpu_memory_gb", memory_allocated)
                
                logger.info("Model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                span.set_attribute("error", True)
                span.record_exception(e)
                self.metrics.record_error("model_load_failed")
                raise RuntimeError(f"Model loading failed: {e}") from e
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with comprehensive instrumentation.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Enable sampling (vs greedy)
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary with generated text and metrics
        
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with self.tracer.start_as_current_span("generate") as span:
            span.set_attribute("prompt.length", len(prompt))
            span.set_attribute("generation.max_new_tokens", max_new_tokens)
            span.set_attribute("generation.temperature", temperature)
            
            try:
                self.metrics.increment_active_requests()
                
                # Tokenize input
                with self.tracer.start_as_current_span("tokenize_input") as tokenize_span:
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    prompt_tokens = inputs["input_ids"].shape[1]
                    
                    tokenize_span.set_attribute("prompt.tokens", prompt_tokens)
                    span.set_attribute("prompt.tokens", prompt_tokens)
                
                # Configure generation
                generation_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
                
                # Generate with timing
                timer = InferenceTimer(self.tracer)
                timer.start("model_generate")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                
                timer.record_first_token()
                timings = timer.stop()
                
                # Decode output
                with self.tracer.start_as_current_span("decode_output"):
                    generated_ids = outputs.sequences[0][prompt_tokens:]
                    generated_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    completion_tokens = len(generated_ids)
                
                # Calculate metrics
                total_tokens = prompt_tokens + completion_tokens
                
                inference_metrics = InferenceMetrics(
                    ttft=timings.first_token_time,
                    tpot=timings.average_tpot,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_time=timings.total_time,
                    memory_used_mb=timer.memory_end_mb or 0.0,
                    gpu_memory_used_mb=timer.gpu_memory_end_mb,
                )
                
                # Record metrics
                self.metrics.record_inference_metrics(inference_metrics)
                
                # Add metrics to span
                span.set_attribute("generation.completion_tokens", completion_tokens)
                span.set_attribute("generation.total_tokens", total_tokens)
                span.set_attribute("metrics.ttft", timings.first_token_time)
                span.set_attribute("metrics.tpot", timings.average_tpot)
                
                logger.info(
                    f"Generation completed - Prompt: {prompt_tokens} tokens, "
                    f"Generated: {completion_tokens} tokens, "
                    f"TTFT: {timings.first_token_time:.3f}s, "
                    f"TPOT: {timings.average_tpot:.3f}s"
                )
                
                return {
                    "generated_text": generated_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "ttft": timings.first_token_time,
                    "tpot": timings.average_tpot,
                    "total_time": timings.total_time,
                    "throughput_tokens_per_sec": total_tokens / timings.total_time,
                }
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error("Out of memory error during generation")
                span.set_attribute("error", True)
                span.record_exception(e)
                self.metrics.record_error("oom")
                raise
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                span.set_attribute("error", True)
                span.record_exception(e)
                self.metrics.record_error("generation_failed")
                raise
                
            finally:
                self.metrics.decrement_active_requests()
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._model_loaded = False
        self.metrics.set_model_loaded(False)
        
        logger.info("Model unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._model_loaded,
            "quantization": self.load_in_8bit,
            "max_length": self.max_length,
        }
        
        if torch.cuda.is_available() and self._model_loaded:
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
        
        return info
