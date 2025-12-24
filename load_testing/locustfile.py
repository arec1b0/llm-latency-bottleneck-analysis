"""
Locust Load Testing for LLM Inference API

Comprehensive load testing scenarios to identify latency bottlenecks.
"""

import json
import logging
import random
import time
from typing import List, Dict, Any

from locust import HttpUser, task, between, events
from locust.env import Environment


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample prompts for realistic testing
SAMPLE_PROMPTS: List[str] = [
    "Explain quantum computing in simple terms",
    "Write a short story about a robot learning to paint",
    "What are the key differences between machine learning and deep learning?",
    "Describe the process of photosynthesis step by step",
    "How does blockchain technology work?",
    "Write a haiku about artificial intelligence",
    "Explain the concept of recursion with an example",
    "What are the main causes of climate change?",
    "Describe the structure of a neural network",
    "How do transformers work in NLP?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the basics of quantum mechanics",
    "How does gradient descent optimization work?",
    "What are the advantages of cloud computing?",
    "Describe the SOLID principles in software engineering",
]


# Prompt categories by length
SHORT_PROMPTS: List[str] = [
    "Define AI",
    "What is ML?",
    "Explain NLP",
    "GPU vs CPU",
    "Docker basics",
]

MEDIUM_PROMPTS: List[str] = SAMPLE_PROMPTS

LONG_PROMPTS: List[str] = [
    "Provide a comprehensive explanation of how large language models work, including the architecture, training process, inference mechanism, and practical applications in production environments.",
    "Describe in detail the differences between various machine learning algorithms including decision trees, random forests, support vector machines, neural networks, and deep learning models, with examples of when to use each approach.",
    "Explain the complete software development lifecycle including requirements gathering, design, implementation, testing, deployment, and maintenance phases, with best practices for each stage.",
]


class PerformanceMetrics:
    """Track custom performance metrics."""
    
    def __init__(self):
        self.ttft_values: List[float] = []
        self.tpot_values: List[float] = []
        self.token_counts: List[int] = []
        self.throughputs: List[float] = []
    
    def add_result(self, response_data: Dict[str, Any]) -> None:
        """Add result from API response."""
        if "ttft" in response_data:
            self.ttft_values.append(response_data["ttft"])
        if "tpot" in response_data:
            self.tpot_values.append(response_data["tpot"])
        if "total_tokens" in response_data:
            self.token_counts.append(response_data["total_tokens"])
        if "throughput_tokens_per_sec" in response_data:
            self.throughputs.append(response_data["throughput_tokens_per_sec"])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.ttft_values:
            return {}
        
        return {
            "ttft": {
                "min": min(self.ttft_values),
                "max": max(self.ttft_values),
                "avg": sum(self.ttft_values) / len(self.ttft_values),
                "p50": sorted(self.ttft_values)[len(self.ttft_values) // 2],
                "p95": sorted(self.ttft_values)[int(len(self.ttft_values) * 0.95)],
                "p99": sorted(self.ttft_values)[int(len(self.ttft_values) * 0.99)],
            },
            "tpot": {
                "min": min(self.tpot_values),
                "max": max(self.tpot_values),
                "avg": sum(self.tpot_values) / len(self.tpot_values),
            },
            "tokens": {
                "min": min(self.token_counts),
                "max": max(self.token_counts),
                "avg": sum(self.token_counts) / len(self.token_counts),
                "total": sum(self.token_counts),
            },
            "throughput": {
                "min": min(self.throughputs),
                "max": max(self.throughputs),
                "avg": sum(self.throughputs) / len(self.throughputs),
            },
            "total_requests": len(self.ttft_values),
        }


# Global metrics collector
performance_metrics = PerformanceMetrics()


class LLMInferenceUser(HttpUser):
    """
    Simulates a user making requests to the LLM inference API.
    
    Configurable wait time between requests and multiple task scenarios.
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts."""
        logger.info(f"User {self.environment.runner.user_count} started")
        
        # Check health on startup
        response = self.client.get("/health")
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"API Health: {health_data}")
    
    @task(10)
    def generate_text_short(self):
        """
        Generate text with short prompts (high frequency).
        
        Simulates quick queries with fast responses.
        """
        prompt = random.choice(SHORT_PROMPTS)
        payload = {
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0.7,
            "do_sample": True,
        }
        
        with self.client.post(
            "/generate",
            json=payload,
            catch_response=True,
            name="/generate [short]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                performance_metrics.add_result(data)
                response.success()
                logger.debug(f"Short generation: TTFT={data['ttft']:.3f}s, Tokens={data['total_tokens']}")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(5)
    def generate_text_medium(self):
        """
        Generate text with medium prompts (medium frequency).
        
        Simulates typical user queries.
        """
        prompt = random.choice(MEDIUM_PROMPTS)
        payload = {
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.7,
            "do_sample": True,
        }
        
        with self.client.post(
            "/generate",
            json=payload,
            catch_response=True,
            name="/generate [medium]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                performance_metrics.add_result(data)
                response.success()
                logger.debug(f"Medium generation: TTFT={data['ttft']:.3f}s, Tokens={data['total_tokens']}")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def generate_text_long(self):
        """
        Generate text with long prompts (low frequency).
        
        Simulates complex, long-form queries that stress the system.
        """
        prompt = random.choice(LONG_PROMPTS)
        payload = {
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7,
            "do_sample": True,
        }
        
        with self.client.post(
            "/generate",
            json=payload,
            catch_response=True,
            name="/generate [long]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                performance_metrics.add_result(data)
                response.success()
                logger.info(f"Long generation: TTFT={data['ttft']:.3f}s, Tokens={data['total_tokens']}")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def generate_text_high_temperature(self):
        """
        Generate text with high temperature (creative mode).
        
        Tests performance with more random sampling.
        """
        prompt = random.choice(MEDIUM_PROMPTS)
        payload = {
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 1.2,
            "top_p": 0.95,
            "do_sample": True,
        }
        
        with self.client.post(
            "/generate",
            json=payload,
            catch_response=True,
            name="/generate [high_temp]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                performance_metrics.add_result(data)
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def check_health(self):
        """
        Health check endpoint (low frequency).
        
        Monitors API availability during load testing.
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def get_model_info(self):
        """
        Get model information (low frequency).
        
        Tests metadata endpoints under load.
        """
        with self.client.get("/model/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model info failed: {response.status_code}")


class HighLoadUser(LLMInferenceUser):
    """
    Aggressive load testing user with minimal wait time.
    
    Used for stress testing and finding breaking points.
    """
    
    wait_time = between(0.1, 0.5)  # Very short wait time


class BurstLoadUser(LLMInferenceUser):
    """
    Burst load user - sends requests in bursts.
    
    Simulates traffic spikes.
    """
    
    wait_time = between(0, 10)  # Either no wait or long wait
    
    def on_start(self):
        super().on_start()
        # Randomly decide if this user will be bursty
        self.is_burst = random.random() < 0.5
    
    @task
    def burst_requests(self):
        """Send multiple requests in quick succession."""
        if self.is_burst:
            for _ in range(random.randint(3, 7)):
                self.generate_text_medium()
                time.sleep(0.1)
            time.sleep(random.randint(5, 15))


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """
    Called when the load test stops.
    
    Prints summary statistics.
    """
    logger.info("=" * 80)
    logger.info("LOAD TEST COMPLETED")
    logger.info("=" * 80)
    
    summary = performance_metrics.get_summary()
    
    if summary:
        logger.info("\n📊 Performance Metrics Summary:")
        logger.info(f"\n⏱️  TTFT (Time to First Token):")
        logger.info(f"   Min:  {summary['ttft']['min']:.3f}s")
        logger.info(f"   Avg:  {summary['ttft']['avg']:.3f}s")
        logger.info(f"   Max:  {summary['ttft']['max']:.3f}s")
        logger.info(f"   P50:  {summary['ttft']['p50']:.3f}s")
        logger.info(f"   P95:  {summary['ttft']['p95']:.3f}s")
        logger.info(f"   P99:  {summary['ttft']['p99']:.3f}s")
        
        logger.info(f"\n🔄 TPOT (Time Per Output Token):")
        logger.info(f"   Min:  {summary['tpot']['min']:.3f}s")
        logger.info(f"   Avg:  {summary['tpot']['avg']:.3f}s")
        logger.info(f"   Max:  {summary['tpot']['max']:.3f}s")
        
        logger.info(f"\n📝 Tokens:")
        logger.info(f"   Min:    {summary['tokens']['min']}")
        logger.info(f"   Avg:    {summary['tokens']['avg']:.1f}")
        logger.info(f"   Max:    {summary['tokens']['max']}")
        logger.info(f"   Total:  {summary['tokens']['total']}")
        
        logger.info(f"\n🚀 Throughput:")
        logger.info(f"   Min:  {summary['throughput']['min']:.2f} tokens/s")
        logger.info(f"   Avg:  {summary['throughput']['avg']:.2f} tokens/s")
        logger.info(f"   Max:  {summary['throughput']['max']:.2f} tokens/s")
        
        logger.info(f"\n📈 Total Requests: {summary['total_requests']}")
        logger.info("=" * 80)
        
        # Save results to file
        try:
            import os
            os.makedirs("load_testing/results", exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"load_testing/results/metrics_{timestamp}.json"
            
            with open(filename, "w") as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
