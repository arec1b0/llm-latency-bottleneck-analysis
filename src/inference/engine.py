"""
Inference Engine

LLM inference engine with comprehensive performance instrumentation.
Loads model, manages generation, and tracks detailed timing metrics.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Iterator, Generator
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    TextStreamer,
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
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate text with streaming output and accurate TTFT/TPOT measurement.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Enable sampling (vs greedy)
            **kwargs: Additional generation parameters
        
        Yields:
            Dictionary with token text and timing metrics
        
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with self.tracer.start_as_current_span("generate_stream") as span:
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
                
                # Start timing
                timer = InferenceTimer(self.tracer)
                timer.start("model_generate_stream")
                
                # Generate with streaming
                with torch.no_grad():
                    generated_ids = inputs["input_ids"].clone()
                    first_token_recorded = False
                    
                    for i in range(max_new_tokens):
                        # Get next token
                        outputs = self.model(
                            input_ids=generated_ids,
                            attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                            use_cache=True,
                        )
                        
                        # Get logits and sample next token
                        logits = outputs.logits[:, -1, :]
                        if do_sample:
                            # Apply temperature
                            logits = logits / temperature
                            # Apply top-p filtering
                            if top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., -1] = 0
                                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                                logits[indices_to_remove] = float('-inf')
                            
                            # Sample
                            probs = torch.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            # Greedy
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        
                        # Record first token time
                        if not first_token_recorded:
                            timer.record_first_token()
                            first_token_recorded = True
                        
                        # Append to sequence
                        generated_ids = torch.cat([generated_ids, next_token], dim=1)
                        
                        # Decode the new token
                        new_token_text = self.tokenizer.decode(
                            next_token[0], 
                            skip_special_tokens=True
                        )
                        
                        # Check for end of sequence
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                        
                        # Yield token with timing
                        current_time = time.perf_counter()
                        yield {
                            "token": new_token_text,
                            "token_id": next_token.item(),
                            "sequence_position": i + 1,
                            "current_time": current_time,
                        }
                
                # Final timing
                timings = timer.stop()
                
                # Decode full sequence for final response
                full_generated_ids = generated_ids[0][prompt_tokens:]
                full_text = self.tokenizer.decode(
                    full_generated_ids,
                    skip_special_tokens=True,
                )
                completion_tokens = len(full_generated_ids)
                
                # Yield final completion info
                yield {
                    "type": "completion",
                    "generated_text": full_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "ttft": timings.first_token_time,
                    "tpot": timings.average_tpot,
                    "total_time": timings.total_time,
                    "throughput_tokens_per_sec": (prompt_tokens + completion_tokens) / timings.total_time,
                }
                
                # Record final metrics
                inference_metrics = InferenceMetrics(
                    ttft=timings.first_token_time,
                    tpot=timings.average_tpot,
                    total_tokens=prompt_tokens + completion_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_time=timings.total_time,
                    memory_used_mb=timer.memory_end_mb or 0.0,
                    gpu_memory_used_mb=timer.gpu_memory_end_mb,
                )
                
                self.metrics.record_inference_metrics(inference_metrics)
                
                logger.info(
                    f"Streaming generation completed - Prompt: {prompt_tokens} tokens, "
                    f"Generated: {completion_tokens} tokens, "
                    f"TTFT: {timings.first_token_time:.3f}s, "
                    f"TPOT: {timings.average_tpot:.3f}s"
                )
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error("Out of memory error during streaming generation")
                span.set_attribute("error", True)
                span.record_exception(e)
                self.metrics.record_error("oom")
                raise
                
            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")
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
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts in batch for improved throughput.
        
        Args:
            prompts: List of input prompt texts
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Enable sampling (vs greedy)
            **kwargs: Additional generation parameters
        
        Returns:
            List of dictionaries with generated text and metrics
        
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with self.tracer.start_as_current_span("generate_batch") as span:
            span.set_attribute("batch.size", len(prompts))
            span.set_attribute("generation.max_new_tokens", max_new_tokens)
            span.set_attribute("generation.temperature", temperature)
            
            try:
                self.metrics.increment_active_requests()
                
                # Tokenize all inputs in batch
                with self.tracer.start_as_current_span("tokenize_batch") as tokenize_span:
                    inputs = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
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
                
                # Start timing
                timer = InferenceTimer(self.tracer)
                timer.start("model_generate_batch")
                
                # Generate in batch
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
                
                # Stop timing
                timings = timer.stop()
                
                # Process results
                results = []
                for i, prompt in enumerate(prompts):
                    # Decode generated text for this prompt
                    generated_ids = outputs.sequences[i]
                    prompt_length = inputs["attention_mask"][i].sum().item()
                    generated_ids_only = generated_ids[prompt_length:]
                    
                    generated_text = self.tokenizer.decode(
                        generated_ids_only,
                        skip_special_tokens=True,
                    )
                    
                    completion_tokens = len(generated_ids_only)
                    total_tokens = prompt_length + completion_tokens
                    
                    # Calculate metrics
                    result = {
                        "generated_text": generated_text,
                        "prompt_tokens": prompt_length,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "ttft": timings.first_token_time,
                        "tpot": timings.average_tpot,
                        "total_time": timings.total_time,
                        "throughput_tokens_per_sec": total_tokens / timings.total_time,
                    }
                    results.append(result)
                
                # Record batch metrics
                total_tokens = sum(r["total_tokens"] for r in results)
                avg_ttft = sum(r["ttft"] for r in results) / len(results)
                avg_tpot = sum(r["tpot"] for r in results) / len(results)
                
                batch_metrics = InferenceMetrics(
                    ttft=avg_ttft,
                    tpot=avg_tpot,
                    total_tokens=total_tokens,
                    prompt_tokens=sum(r["prompt_tokens"] for r in results),
                    completion_tokens=sum(r["completion_tokens"] for r in results),
                    total_time=timings.total_time,
                    memory_used_mb=timer.memory_end_mb or 0.0,
                    gpu_memory_used_mb=timer.gpu_memory_end_mb,
                )
                
                self.metrics.record_inference_metrics(batch_metrics)
                
                logger.info(
                    f"Batch generation completed - Batch size: {len(prompts)}, "
                    f"Total tokens: {total_tokens}, "
                    f"Avg TTFT: {avg_ttft:.3f}s, "
                    f"Avg TPOT: {avg_tpot:.3f}s"
                )
                
                return results
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error("Out of memory error during batch generation")
                span.set_attribute("error", True)
                span.record_exception(e)
                self.metrics.record_error("oom")
                raise
                
            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                span.set_attribute("error", True)
                span.record_exception(e)
                self.metrics.record_error("generation_failed")
                raise
                
            finally:
                self.metrics.decrement_active_requests()

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate text with streaming output and comprehensive metrics.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Yields:
            Dict containing:
            - token: Generated token
            - text: Accumulated generated text
            - finished: Whether generation is complete
            - metrics: Current generation metrics
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        self.metrics.increment_active_requests()
        start_time = time.time()
        
        with self.tracer.start_as_current_span("generate_stream") as span:
            try:
                # Set span attributes
                span.set_attribute("llm.prompt_length", len(prompt))
                span.set_attribute("llm.max_new_tokens", max_new_tokens)
                span.set_attribute("llm.temperature", temperature)
                span.set_attribute("llm.top_p", top_p)
                
                # Tokenize input
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generation config
                generation_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
                
                # Custom streamer for token-by-token generation
                class MetricsStreamer(TextStreamer):
                    def __init__(self, tokenizer, metrics_collector, tracer, span):
                        super().__init__(tokenizer, skip_prompt=True)
                        self.metrics = metrics_collector
                        self.tracer = tracer
                        self.span = span
                        self.token_times = []
                        self.first_token_time = None
                        self.start_time = time.time()
                        
                    def on_finalized_text(self, text: str, stream_end: bool = False):
                        current_time = time.time()
                        
                        if self.first_token_time is None:
                            self.first_token_time = current_time - self.start_time
                            self.span.set_attribute("llm.first_token_time", self.first_token_time)
                            
                        self.token_times.append(current_time)
                        
                        # Calculate TPOT for tokens after the first
                        if len(self.token_times) > 1:
                            recent_times = self.token_times[-10:]  # Last 10 tokens
                            tpot = sum(recent_times[i] - recent_times[i-1] for i in range(1, len(recent_times))) / (len(recent_times) - 1)
                            self.span.set_attribute("llm.tpot", tpot)
                
                # Create streamer
                streamer = MetricsStreamer(
                    self.tokenizer, 
                    self.metrics, 
                    self.tracer, 
                    span
                )
                
                # Generate with streaming
                accumulated_text = ""
                token_count = 0
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        streamer=streamer
                    )
                
                # Extract generated sequence
                generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
                
                # Yield tokens one by one
                for i, token_id in enumerate(generated_ids):
                    token_text = self.tokenizer.decode(token_id, skip_special_tokens=True)
                    accumulated_text += token_text
                    token_count += 1
                    
                    # Calculate metrics
                    current_time = time.time()
                    total_time = current_time - start_time
                    ttft = streamer.first_token_time if streamer.first_token_time else total_time
                    tpot = (total_time - ttft) / max(token_count - 1, 1) if token_count > 1 else 0
                    throughput = token_count / total_time if total_time > 0 else 0
                    
                    yield {
                        "token": token_text,
                        "text": accumulated_text,
                        "finished": i == len(generated_ids) - 1,
                        "metrics": {
                            "token_index": i,
                            "total_tokens": token_count,
                            "ttft": ttft,
                            "tpot": tpot,
                            "total_time": total_time,
                            "throughput_tokens_per_sec": throughput,
                            "prompt_tokens": inputs["input_ids"].shape[1],
                            "completion_tokens": token_count,
                        }
                    }
                
                # Record final metrics
                total_time = time.time() - start_time
                final_metrics = {
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "completion_tokens": len(generated_ids),
                    "total_tokens": inputs["input_ids"].shape[1] + len(generated_ids),
                    "ttft": streamer.first_token_time if streamer.first_token_time else total_time,
                    "tpot": (total_time - (streamer.first_token_time or 0)) / max(len(generated_ids) - 1, 1),
                    "total_time": total_time,
                    "throughput_tokens_per_sec": len(generated_ids) / total_time if total_time > 0 else 0,
                }
                
                # Record metrics
                self.metrics.record_inference_metrics(
                    prompt_tokens=final_metrics["prompt_tokens"],
                    completion_tokens=final_metrics["completion_tokens"],
                    total_time=final_metrics["total_time"],
                    ttft=final_metrics["ttft"],
                    tpot=final_metrics["tpot"]
                )
                
                # Set final span attributes
                span.set_attribute("llm.prompt_tokens", final_metrics["prompt_tokens"])
                span.set_attribute("llm.completion_tokens", final_metrics["completion_tokens"])
                span.set_attribute("llm.total_time", final_metrics["total_time"])
                span.set_attribute("llm.throughput", final_metrics["throughput_tokens_per_sec"])
                
                logger.info(
                    f"Streaming generation completed: "
                    f"{final_metrics['completion_tokens']} tokens in "
                    f"{final_metrics['total_time']:.2f}s "
                    f"(TTFT: {final_metrics['ttft']:.3f}s, "
                    f"TPOT: {final_metrics['tpot']:.3f}s)"
                )
                
            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")
                span.record_exception(e)
                self.metrics.record_error("streaming_generation_failed")
                raise
                
            finally:
                self.metrics.decrement_active_requests()
