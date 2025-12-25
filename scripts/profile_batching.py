#!/usr/bin/env python3
"""
Batch vs Single Generation Performance Profiler

Compares performance between single and batch generation modes.
Measures throughput, latency, and resource utilization.
"""

import asyncio
import json
import statistics
import time
from typing import List, Dict, Any
import httpx
import click


class BatchProfiler:
    """Profiler for comparing batch vs single generation performance."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def single_generate(self, prompt: str) -> Dict[str, Any]:
        """Generate text for a single prompt."""
        response = await self.client.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def batch_generate(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate text for multiple prompts in batch."""
        response = await self.client.post(
            f"{self.base_url}/generate_batch",
            json={
                "prompts": prompts,
                "max_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def profile_single(
        self, prompts: List[str], concurrent: int = 1
    ) -> Dict[str, Any]:
        """Profile single generation with specified concurrency."""
        print(f"Profiling single generation (concurrency: {concurrent})...")
        
        start_time = time.time()
        results = []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent)
        
        async def generate_single(prompt: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.single_generate(prompt)
        
        # Run all requests
        tasks = [generate_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        total_tokens = sum(r["total_tokens"] for r in results)
        ttfts = [r["ttft"] for r in results]
        tpots = [r["tpot"] for r in results]
        
        return {
            "mode": "single",
            "concurrency": concurrent,
            "num_requests": len(prompts),
            "total_time": total_time,
            "total_tokens": total_tokens,
            "throughput_tokens_per_sec": total_tokens / total_time,
            "requests_per_sec": len(prompts) / total_time,
            "avg_ttft": statistics.mean(ttfts),
            "avg_tpot": statistics.mean(tpots),
            "median_ttft": statistics.median(ttfts),
            "median_tpot": statistics.median(tpots),
            "p95_ttft": sorted(ttfts)[int(len(ttfts) * 0.95)],
            "p95_tpot": sorted(tpots)[int(len(tpots) * 0.95)],
        }
    
    async def profile_batch(
        self, prompts: List[str], batch_sizes: List[int]
    ) -> List[Dict[str, Any]]:
        """Profile batch generation with different batch sizes."""
        results = []
        
        for batch_size in batch_sizes:
            print(f"Profiling batch generation (batch size: {batch_size})...")
            
            # Split prompts into batches
            batches = [
                prompts[i:i + batch_size] 
                for i in range(0, len(prompts), batch_size)
            ]
            
            start_time = time.time()
            batch_results = []
            
            for batch in batches:
                result = await self.batch_generate(batch)
                batch_results.append(result)
            
            total_time = time.time() - start_time
            
            # Aggregate metrics
            total_tokens = sum(r["total_tokens"] for r in batch_results)
            total_requests = sum(r["batch_size"] for r in batch_results)
            
            # Collect individual TTFT/TPOT from all results
            all_ttfts = []
            all_tpots = []
            for batch_result in batch_results:
                for result in batch_result["results"]:
                    all_ttfts.append(result["ttft"])
                    all_tpots.append(result["tpot"])
            
            results.append({
                "mode": "batch",
                "batch_size": batch_size,
                "num_batches": len(batches),
                "num_requests": total_requests,
                "total_time": total_time,
                "total_tokens": total_tokens,
                "throughput_tokens_per_sec": total_tokens / total_time,
                "requests_per_sec": total_requests / total_time,
                "avg_ttft": statistics.mean(all_ttfts),
                "avg_tpot": statistics.mean(all_tpots),
                "median_ttft": statistics.median(all_ttfts),
                "median_tpot": statistics.median(all_tpots),
                "p95_ttft": sorted(all_ttfts)[int(len(all_ttfts) * 0.95)],
                "p95_tpot": sorted(all_tpots)[int(len(all_tpots) * 0.95)],
            })
        
        return results
    
    async def run_profile(
        self, 
        num_requests: int = 50,
        batch_sizes: List[int] = [4, 8, 16, 32],
        concurrency_levels: List[int] = [1, 2, 4, 8],
    ) -> Dict[str, Any]:
        """Run complete profiling comparison."""
        print("Starting batch vs single generation profiling...")
        print(f"Total requests: {num_requests}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Concurrency levels: {concurrency_levels}")
        print()
        
        # Generate test prompts
        prompts = [
            f"The future of artificial intelligence is {i}. "
            f"Please elaborate on this topic with specific examples and insights."
            for i in range(num_requests)
        ]
        
        # Profile single generation with different concurrency
        single_results = []
        for concurrency in concurrency_levels:
            result = await self.profile_single(prompts, concurrency)
            single_results.append(result)
            print(f"Single (concurrency={concurrency}): "
                  f"{result['throughput_tokens_per_sec']:.1f} tokens/sec, "
                  f"{result['requests_per_sec']:.1f} req/sec")
        
        print()
        
        # Profile batch generation
        batch_results = await self.profile_batch(prompts, batch_sizes)
        for result in batch_results:
            print(f"Batch (size={result['batch_size']}): "
                  f"{result['throughput_tokens_per_sec']:.1f} tokens/sec, "
                  f"{result['requests_per_sec']:.1f} req/sec")
        
        print()
        
        # Find best configurations
        best_single = max(single_results, key=lambda x: x["throughput_tokens_per_sec"])
        best_batch = max(batch_results, key=lambda x: x["throughput_tokens_per_sec"])
        
        print("=== Summary ===")
        print(f"Best single: concurrency={best_single['concurrency']}, "
              f"throughput={best_single['throughput_tokens_per_sec']:.1f} tokens/sec")
        print(f"Best batch: size={best_batch['batch_size']}, "
              f"throughput={best_batch['throughput_tokens_per_sec']:.1f} tokens/sec")
        
        improvement = (best_batch['throughput_tokens_per_sec'] / 
                      best_single['throughput_tokens_per_sec'] - 1) * 100
        print(f"Batch improvement: {improvement:.1f}%")
        
        return {
            "single_results": single_results,
            "batch_results": batch_results,
            "best_single": best_single,
            "best_batch": best_batch,
            "improvement_percent": improvement,
        }


@click.command()
@click.option("--url", default="http://localhost:8000", help="API base URL")
@click.option("--requests", default=50, help="Number of test requests")
@click.option("--batch-sizes", default="4,8,16,32", help="Comma-separated batch sizes")
@click.option("--concurrency", default="1,2,4,8", help="Comma-separated concurrency levels")
@click.option("--output", default="batch_profile_results.json", help="Output file")
async def main(url: str, requests: int, batch_sizes: str, concurrency: str, output: str):
    """Run batch vs single generation profiling."""
    
    # Parse options
    batch_size_list = [int(x.strip()) for x in batch_sizes.split(",")]
    concurrency_list = [int(x.strip()) for x in concurrency.split(",")]
    
    async with BatchProfiler(url) as profiler:
        # Check API health
        if not await profiler.health_check():
            print(f"API at {url} is not healthy")
            return
        
        # Run profiling
        results = await profiler.run_profile(
            num_requests=requests,
            batch_sizes=batch_size_list,
            concurrency_levels=concurrency_list,
        )
        
        # Save results
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output}")


if __name__ == "__main__":
    asyncio.run(main())
