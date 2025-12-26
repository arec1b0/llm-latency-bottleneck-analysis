#!/usr/bin/env python3
"""
Comprehensive performance testing script for LLM inference.
Tests single generation, batch generation, streaming, and concurrent requests.
"""

import asyncio
import json
import time
import statistics
from typing import List, Dict, Any
import aiohttp
import argparse
import pandas as pd
from datetime import datetime


class PerformanceTestSuite:
    """Comprehensive performance testing suite for LLM inference."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        
    async def test_batching_performance(
        self,
        prompt: str,
        max_tokens: int = 256,
        concurrency_levels: List[int] = [1, 2, 4, 8],
        requests_per_level: int = 20
    ) -> Dict[str, Any]:
        """Test batching performance with different concurrency levels."""
        print(f"Testing batching performance...")
        
        url = f"{self.base_url}/generate"
        request_data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        results = {
            "concurrency_levels": concurrency_levels,
            "metrics": {}
        }
        
        for concurrency in concurrency_levels:
            print(f"  Testing concurrency level: {concurrency}")
            
            times = []
            ttfts = []
            tpots = []
            throughputs = []
            errors = 0
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                # Create semaphore for concurrency control
                semaphore = asyncio.Semaphore(concurrency)
                
                async def make_request():
                    async with semaphore:
                        try:
                            start_time = time.time()
                            async with session.post(url, json=request_data) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    end_time = time.time()
                                    
                                    times.append(end_time - start_time)
                                    ttfts.append(result.get("ttft", 0))
                                    tpots.append(result.get("tpot", 0))
                                    throughputs.append(result.get("throughput_tokens_per_sec", 0))
                                else:
                                    errors += 1
                                    print(f"    Error: HTTP {response.status}")
                        except Exception as e:
                            errors += 1
                            print(f"    Exception: {e}")
                
                # Run requests concurrently
                tasks = [make_request() for _ in range(requests_per_level)]
                await asyncio.gather(*tasks)
            
            # Calculate metrics for this concurrency level
            if times:
                results["metrics"][concurrency] = {
                    "avg_time": statistics.mean(times),
                    "p95_time": statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times),
                    "avg_ttft": statistics.mean(ttfts) if ttfts else 0,
                    "avg_tpot": statistics.mean(tpots) if tpots else 0,
                    "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
                    "total_requests": requests_per_level,
                    "successful_requests": len(times),
                    "error_rate": errors / requests_per_level,
                    "requests_per_second": len(times) / sum(times) if times else 0
                }
            else:
                results["metrics"][concurrency] = {
                    "avg_time": 0,
                    "p95_time": 0,
                    "avg_ttft": 0,
                    "avg_tpot": 0,
                    "avg_throughput": 0,
                    "total_requests": requests_per_level,
                    "successful_requests": 0,
                    "error_rate": 1.0,
                    "requests_per_second": 0
                }
            
            print(f"    Avg time: {results['metrics'][concurrency]['avg_time']:.3f}s")
            print(f"    Error rate: {results['metrics'][concurrency]['error_rate']:.2%}")
            print(f"    Throughput: {results['metrics'][concurrency]['avg_throughput']:.1f} tok/s")
        
        return results
    
    async def test_single_generation(
        self,
        prompt: str,
        max_tokens: int = 256,
        num_requests: int = 10
    ) -> Dict[str, Any]:
        """Test single generation performance."""
        print(f"Testing single generation ({num_requests} requests)...")
        
        url = f"{self.base_url}/generate"
        request_data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        times = []
        ttfts = []
        tpots = []
        throughputs = []
        tokens_generated = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                start_time = time.time()
                async with session.post(url, json=request_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Request {i+1} failed: HTTP {response.status}: {error_text}")
                    
                    result = await response.json()
                    total_time = time.time() - start_time
                    
                    times.append(total_time)
                    ttfts.append(result.get("ttft", 0))
                    tpots.append(result.get("tpot", 0))
                    throughputs.append(result.get("throughput_tokens_per_sec", 0))
                    tokens_generated.append(result.get("completion_tokens", 0))
                
                if (i + 1) % 5 == 0:
                    print(f"  Completed {i+1}/{num_requests} requests")
        
        # Calculate statistics
        return {
            "type": "single_generation",
            "num_requests": num_requests,
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "median_time": statistics.median(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_ttft": statistics.mean(ttfts),
            "avg_tpot": statistics.mean(tpots),
            "avg_throughput": statistics.mean(throughputs),
            "total_tokens": sum(tokens_generated),
            "avg_tokens_per_request": statistics.mean(tokens_generated),
            "requests_per_second": num_requests / sum(times),
            "raw_times": times,
            "raw_ttfts": ttfts,
            "raw_tpots": tpots,
            "raw_throughputs": throughputs
        }
    
    async def test_batch_generation(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        num_batches: int = 10
    ) -> Dict[str, Any]:
        """Test batch generation performance."""
        print(f"Testing batch generation (batch_size={len(prompts)}, {num_batches} batches)...")
        
        url = f"{self.base_url}/generate_batch"
        request_data = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        times = []
        total_tokens_per_batch = []
        throughputs_per_batch = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_batches):
                start_time = time.time()
                async with session.post(url, json=request_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Batch {i+1} failed: HTTP {response.status}: {error_text}")
                    
                    result = await response.json()
                    total_time = time.time() - start_time
                    
                    times.append(total_time)
                    total_tokens_per_batch.append(result.get("total_tokens", 0))
                    throughputs_per_batch.append(result.get("throughput_tokens_per_sec", 0))
                
                if (i + 1) % 3 == 0:
                    print(f"  Completed {i+1}/{num_batches} batches")
        
        # Calculate statistics
        total_requests = num_batches * len(prompts)
        return {
            "type": "batch_generation",
            "batch_size": len(prompts),
            "num_batches": num_batches,
            "total_requests": total_requests,
            "avg_time_per_batch": statistics.mean(times),
            "min_time_per_batch": min(times),
            "max_time_per_batch": max(times),
            "median_time_per_batch": statistics.median(times),
            "std_time_per_batch": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_throughput_per_batch": statistics.mean(throughputs_per_batch),
            "total_tokens": sum(total_tokens_per_batch),
            "avg_tokens_per_batch": statistics.mean(total_tokens_per_batch),
            "avg_time_per_request": statistics.mean(times) / len(prompts),
            "requests_per_second": total_requests / sum(times),
            "raw_times": times,
            "raw_total_tokens": total_tokens_per_batch,
            "raw_throughputs": throughputs_per_batch
        }
    
    async def test_streaming_generation(
        self,
        prompt: str,
        max_tokens: int = 256,
        num_requests: int = 10
    ) -> Dict[str, Any]:
        """Test streaming generation performance."""
        print(f"Testing streaming generation ({num_requests} requests)...")
        
        url = f"{self.base_url}/generate_stream"
        request_data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        times = []
        first_token_times = []
        throughputs = []
        tokens_generated = []
        
        async def stream_request():
            start_time = time.time()
            first_token_time = None
            token_count = 0
            accumulated_text = ""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                token = data.get("token", "")
                                finished = data.get("finished", False)
                                metrics = data.get("metrics", {})
                                
                                if token and first_token_time is None:
                                    first_token_time = time.time() - start_time
                                
                                if token:
                                    token_count += 1
                                    accumulated_text += token
                                
                                if finished:
                                    break
                            except json.JSONDecodeError:
                                continue
            
            total_time = time.time() - start_time
            return {
                "total_time": total_time,
                "first_token_time": first_token_time or total_time,
                "token_count": token_count,
                "throughput": token_count / total_time if total_time > 0 else 0,
                "accumulated_text": accumulated_text
            }
        
        # Run requests sequentially to avoid overwhelming the server
        for i in range(num_requests):
            result = await stream_request()
            times.append(result["total_time"])
            first_token_times.append(result["first_token_time"])
            throughputs.append(result["throughput"])
            tokens_generated.append(result["token_count"])
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{num_requests} requests")
        
        return {
            "type": "streaming_generation",
            "num_requests": num_requests,
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "median_time": statistics.median(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_first_token_time": statistics.mean(first_token_times),
            "avg_throughput": statistics.mean(throughputs),
            "total_tokens": sum(tokens_generated),
            "avg_tokens_per_request": statistics.mean(tokens_generated),
            "requests_per_second": num_requests / sum(times),
            "raw_times": times,
            "raw_first_token_times": first_token_times,
            "raw_throughputs": throughputs
        }
    
    async def test_concurrent_requests(
        self,
        prompt: str,
        max_tokens: int = 256,
        concurrency_levels: List[int] = [1, 2, 4, 8, 16]
    ) -> Dict[str, Any]:
        """Test performance under different concurrency levels."""
        print("Testing concurrent requests...")
        
        url = f"{self.base_url}/generate"
        request_data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        concurrency_results = {}
        
        for concurrency in concurrency_levels:
            print(f"  Testing concurrency level: {concurrency}")
            
            async def single_request():
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=request_data) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"HTTP {response.status}: {error_text}")
                        await response.json()
                return time.time() - start_time
            
            # Run concurrent requests
            start_time = time.time()
            tasks = [single_request() for _ in range(concurrency)]
            times = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            concurrency_results[concurrency] = {
                "concurrency": concurrency,
                "total_time": total_time,
                "avg_individual_time": statistics.mean(times),
                "min_individual_time": min(times),
                "max_individual_time": max(times),
                "std_individual_time": statistics.stdev(times) if len(times) > 1 else 0,
                "requests_per_second": concurrency / total_time,
                "raw_times": times
            }
        
        return {
            "type": "concurrent_requests",
            "concurrency_levels": concurrency_levels,
            "results": concurrency_results
        }
    
    async def run_comprehensive_test(
        self,
        prompt: str = "The future of artificial intelligence is",
        max_tokens: int = 128,
        batch_sizes: List[int] = [1, 4, 8, 16],
        num_requests: int = 20
    ) -> Dict[str, Any]:
        """Run comprehensive performance test suite."""
        print("=" * 80)
        print("COMPREHENSIVE LLM INFERENCE PERFORMANCE TEST")
        print("=" * 80)
        print(f"Prompt: '{prompt}'")
        print(f"Max tokens: {max_tokens}")
        print(f"Number of requests: {num_requests}")
        print(f"Batch sizes: {batch_sizes}")
        print("-" * 80)
        
        # Test single generation
        single_results = await self.test_single_generation(prompt, max_tokens, num_requests)
        
        # Test batch generation for different batch sizes
        batch_results = {}
        for batch_size in batch_sizes:
            if batch_size == 1:
                batch_results[batch_size] = single_results  # Already tested
            else:
                prompts = [prompt] * batch_size
                num_batches = max(1, num_requests // batch_size)
                batch_results[batch_size] = await self.test_batch_generation(prompts, max_tokens, num_batches)
        
        # Test streaming generation
        streaming_results = await self.test_streaming_generation(prompt, max_tokens, num_requests)
        
        # Test concurrent requests
        concurrent_results = await self.test_concurrent_requests(prompt, max_tokens)
        
        # Compile all results
        all_results = {
            "test_config": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "num_requests": num_requests,
                "batch_sizes": batch_sizes,
                "timestamp": datetime.now().isoformat()
            },
            "single_generation": single_results,
            "batch_generation": batch_results,
            "streaming_generation": streaming_results,
            "concurrent_requests": concurrent_results
        }
        
        # Generate summary
        summary = self.generate_summary(all_results)
        all_results["summary"] = summary
        
        return all_results
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary and comparisons."""
        summary = {
            "key_findings": [],
            "performance_comparison": {},
            "recommendations": []
        }
        
        single = results["single_generation"]
        streaming = results["streaming_generation"]
        batch = results["batch_generation"]
        concurrent = results["concurrent_requests"]
        
        # Single vs Streaming comparison
        time_diff = streaming["avg_time"] - single["avg_time"]
        throughput_diff = streaming["avg_throughput"] - single["avg_throughput"]
        
        summary["performance_comparison"]["single_vs_streaming"] = {
            "time_difference_ms": time_diff * 1000,
            "time_difference_percent": (time_diff / single["avg_time"]) * 100,
            "throughput_difference": throughput_diff,
            "throughput_difference_percent": (throughput_diff / single["avg_throughput"]) * 100
        }
        
        # Batch efficiency analysis
        batch_efficiency = {}
        baseline_throughput = single["avg_throughput"]
        
        for batch_size, batch_result in batch.items():
            if batch_size > 1:
                efficiency = batch_result["avg_throughput_per_batch"] / (baseline_throughput * batch_size)
                batch_efficiency[batch_size] = efficiency
        
        summary["performance_comparison"]["batch_efficiency"] = batch_efficiency
        
        # Concurrency scaling
        concurrency_scaling = {}
        baseline_rps = single["requests_per_second"]
        
        for level, result in concurrent["results"].items():
            if level > 1:
                scaling = result["requests_per_second"] / (baseline_rps * level)
                concurrency_scaling[level] = scaling
        
        summary["performance_comparison"]["concurrency_scaling"] = concurrency_scaling
        
        # Key findings
        if streaming["avg_first_token_time"] < single["avg_ttft"]:
            summary["key_findings"].append("Streaming shows better first token latency")
        
        best_batch_size = max(batch_efficiency.items(), key=lambda x: x[1]) if batch_efficiency else (1, 1.0)
        summary["key_findings"].append(f"Best batch size for efficiency: {best_batch_size[0]} (efficiency: {best_batch_size[1]:.2f})")
        
        best_concurrency = max(concurrency_scaling.items(), key=lambda x: x[1]) if concurrency_scaling else (1, 1.0)
        summary["key_findings"].append(f"Best concurrency level: {best_concurrency[0]} (scaling: {best_concurrency[1]:.2f})")
        
        # Recommendations
        if streaming["avg_first_token_time"] < single["avg_ttft"] * 0.8:
            summary["recommendations"].append("Use streaming for better user experience")
        
        if best_batch_size[0] > 1 and best_batch_size[1] > 1.2:
            summary["recommendations"].append(f"Use batch size {best_batch_size[0]} for improved throughput")
        
        if best_concurrency[0] > 1 and best_concurrency[1] > 0.8:
            summary["recommendations"].append(f"Support concurrency up to {best_concurrency[0]} requests")
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {filename}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print performance summary."""
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 80)
        
        summary = results.get("summary", {})
        
        print("\nKEY FINDINGS:")
        for finding in summary.get("key_findings", []):
            print(f"  • {finding}")
        
        print("\nPERFORMANCE COMPARISONS:")
        
        # Single vs Streaming
        single_vs_streaming = summary["performance_comparison"].get("single_vs_streaming", {})
        if single_vs_streaming:
            print(f"\nSingle vs Streaming:")
            print(f"  Time difference: {single_vs_streaming['time_difference_ms']:.1f}ms "
                  f"({single_vs_streaming['time_difference_percent']:+.1f}%)")
            print(f"  Throughput difference: {single_vs_streaming['throughput_difference']:+.1f} tok/s "
                  f"({single_vs_streaming['throughput_difference_percent']:+.1f}%)")
        
        # Batch efficiency
        batch_efficiency = summary["performance_comparison"].get("batch_efficiency", {})
        if batch_efficiency:
            print(f"\nBatch Efficiency (higher is better):")
            for batch_size, efficiency in sorted(batch_efficiency.items()):
                print(f"  Batch size {batch_size}: {efficiency:.2f}x")
        
        # Concurrency scaling
        concurrency_scaling = summary["performance_comparison"].get("concurrency_scaling", {})
        if concurrency_scaling:
            print(f"\nConcurrency Scaling (higher is better):")
            for level, scaling in sorted(concurrency_scaling.items()):
                print(f"  Concurrency {level}: {scaling:.2f}x")
        
        print("\nRECOMMENDATIONS:")
        for rec in summary.get("recommendations", []):
            print(f"  • {rec}")


async def main():
    parser = argparse.ArgumentParser(description="Comprehensive LLM performance testing")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--prompt", default="The future of artificial intelligence is", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--num-requests", type=int, default=20, help="Number of requests for testing")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16], help="Batch sizes to test")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--summary-only", action="store_true", help="Only show summary")
    parser.add_argument("--test-batching", action="store_true", help="Test batching performance")
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 2, 4, 8], help="Concurrency levels for batching test")
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = PerformanceTestSuite(args.url)
    
    try:
        if args.test_batching:
            # Run batching performance test
            print("=== BATCHING PERFORMANCE TEST ===")
            results = await test_suite.test_batching_performance(
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                concurrency_levels=args.concurrency_levels,
                requests_per_level=args.num_requests
            )
            test_suite.results["batching_performance"] = results
            
            # Print batching results
            print("\n=== BATCHING RESULTS ===")
            for level, metrics in results["metrics"].items():
                print(f"Concurrency {level}:")
                print(f"  Avg time: {metrics['avg_time']:.3f}s")
                print(f"  Error rate: {metrics['error_rate']:.2%}")
                print(f"  Throughput: {metrics['avg_throughput']:.1f} tok/s")
                print(f"  Requests/sec: {metrics['requests_per_second']:.1f}")
        else:
            # Run comprehensive test
            results = await test_suite.run_comprehensive_test(
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                batch_sizes=args.batch_sizes,
                num_requests=args.num_requests
            )
        
        # Print results
        if not args.summary_only:
            test_suite.print_summary(results)
        else:
            test_suite.print_summary(results)
        
        # Save results
        if args.output:
            test_suite.save_results(results, args.output)
        
        return 0
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
