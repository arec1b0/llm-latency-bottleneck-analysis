#!/usr/bin/env python3
"""
Test script for streaming LLM generation.
Tests the /generate_stream endpoint and measures performance.
"""

import asyncio
import json
import time
from typing import AsyncGenerator
import aiohttp
import argparse


async def stream_response(session: aiohttp.ClientSession, url: str, request_data: dict) -> AsyncGenerator[dict, None]:
    """
    Stream responses from the API and yield parsed JSON objects.
    """
    async with session.post(url, json=request_data) as response:
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"HTTP {response.status}: {error_text}")
        
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    yield data
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {e}")
                    continue


async def test_streaming(
    base_url: str = "http://localhost:8000",
    prompt: str = "The future of artificial intelligence is",
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> dict:
    """
    Test streaming generation and collect performance metrics.
    """
    url = f"{base_url}/generate_stream"
    request_data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True
    }
    
    print(f"Testing streaming generation...")
    print(f"URL: {url}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}")
    print("-" * 50)
    
    start_time = time.time()
    tokens = []
    metrics_history = []
    first_token_time = None
    total_tokens = 0
    
    async with aiohttp.ClientSession() as session:
        async for token_data in stream_response(session, url, request_data):
            current_time = time.time()
            
            # Record first token time
            if first_token_time is None and "token" in token_data:
                first_token_time = current_time - start_time
            
            # Extract token and metrics
            token = token_data.get("token", "")
            metrics = token_data.get("metrics", {})
            finished = token_data.get("finished", False)
            
            if token:
                tokens.append(token)
                total_tokens += 1
                
                # Print token with timing
                elapsed = current_time - start_time
                print(f"[{elapsed:.3f}s] Token: '{token}'")
                
                if metrics:
                    metrics_history.append(metrics.copy())
                    print(f"  Metrics: TTFT={metrics.get('ttft', 0):.3f}s, "
                          f"TPOT={metrics.get('tpot', 0):.3f}s, "
                          f"Throughput={metrics.get('throughput_tokens_per_sec', 0):.1f} tok/s")
            
            if finished:
                total_time = current_time - start_time
                print(f"\nGeneration completed in {total_time:.3f}s")
                break
    
    # Calculate final metrics
    total_time = time.time() - start_time
    generated_text = "".join(tokens)
    
    final_metrics = {
        "prompt": prompt,
        "generated_text": generated_text,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "first_token_time": first_token_time,
        "average_throughput": total_tokens / total_time if total_time > 0 else 0,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
        "tokens": tokens,
        "metrics_history": metrics_history
    }
    
    if metrics_history:
        final_metrics.update({
            "final_ttft": metrics_history[-1].get("ttft", 0),
            "final_tpot": metrics_history[-1].get("tpot", 0),
            "final_throughput": metrics_history[-1].get("throughput_tokens_per_sec", 0)
        })
    
    return final_metrics


async def compare_streaming_vs_standard(
    base_url: str = "http://localhost:8000",
    prompt: str = "The future of artificial intelligence is",
    max_tokens: int = 50
) -> dict:
    """
    Compare streaming vs standard generation performance.
    """
    print("=" * 60)
    print("COMPARISON: Streaming vs Standard Generation")
    print("=" * 60)
    
    # Test streaming
    print("\n1. Testing Streaming Generation:")
    print("-" * 40)
    streaming_metrics = await test_streaming(base_url, prompt, max_tokens)
    
    # Test standard generation
    print("\n2. Testing Standard Generation:")
    print("-" * 40)
    
    url = f"{base_url}/generate"
    request_data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
    
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=request_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")
            
            result = await response.json()
    
    total_time = time.time() - start_time
    standard_metrics = {
        "generated_text": result.get("generated_text", ""),
        "total_tokens": result.get("completion_tokens", 0),
        "total_time": total_time,
        "ttft": result.get("ttft", 0),
        "tpot": result.get("tpot", 0),
        "throughput": result.get("throughput_tokens_per_sec", 0)
    }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\nStreaming Generation:")
    print(f"  Text: '{streaming_metrics['generated_text']}'")
    print(f"  Tokens: {streaming_metrics['total_tokens']}")
    print(f"  Time: {streaming_metrics['total_time']:.3f}s")
    print(f"  TTFT: {streaming_metrics.get('final_ttft', 0):.3f}s")
    print(f"  TPOT: {streaming_metrics.get('final_tpot', 0):.3f}s")
    print(f"  Throughput: {streaming_metrics.get('final_throughput', 0):.1f} tok/s")
    
    print(f"\nStandard Generation:")
    print(f"  Text: '{standard_metrics['generated_text']}'")
    print(f"  Tokens: {standard_metrics['total_tokens']}")
    print(f"  Time: {standard_metrics['total_time']:.3f}s")
    print(f"  TTFT: {standard_metrics['ttft']:.3f}s")
    print(f"  TPOT: {standard_metrics['tpot']:.3f}s")
    print(f"  Throughput: {standard_metrics['throughput']:.1f} tok/s")
    
    # Calculate differences
    time_diff = streaming_metrics['total_time'] - standard_metrics['total_time']
    throughput_diff = streaming_metrics.get('final_throughput', 0) - standard_metrics['throughput']
    
    print(f"\nDifferences (Streaming - Standard):")
    print(f"  Time difference: {time_diff:+.3f}s ({time_diff/standard_metrics['total_time']*100:+.1f}%)")
    print(f"  Throughput difference: {throughput_diff:+.1f} tok/s ({throughput_diff/standard_metrics['throughput']*100:+.1f}%)")
    
    return {
        "streaming": streaming_metrics,
        "standard": standard_metrics,
        "comparison": {
            "time_difference": time_diff,
            "time_difference_percent": time_diff/standard_metrics['total_time']*100,
            "throughput_difference": throughput_diff,
            "throughput_difference_percent": throughput_diff/standard_metrics['throughput']*100
        }
    }


async def test_concurrent_streaming(
    base_url: str = "http://localhost:8000",
    num_concurrent: int = 5,
    prompt: str = "The future of AI is",
    max_tokens: int = 30
) -> dict:
    """
    Test concurrent streaming requests.
    """
    print(f"\nTesting {num_concurrent} concurrent streaming requests...")
    print("-" * 50)
    
    async def single_request():
        return await test_streaming(base_url, prompt, max_tokens)
    
    start_time = time.time()
    tasks = [single_request() for _ in range(num_concurrent)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Aggregate results
    total_tokens = sum(r['total_tokens'] for r in results)
    avg_throughput = sum(r.get('final_throughput', 0) for r in results) / len(results)
    avg_ttft = sum(r.get('final_ttft', 0) for r in results) / len(results)
    avg_tpot = sum(r.get('final_tpot', 0) for r in results) / len(results)
    
    concurrent_metrics = {
        "num_concurrent": num_concurrent,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "combined_throughput": total_tokens / total_time,
        "average_individual_throughput": avg_throughput,
        "average_ttft": avg_ttft,
        "average_tpot": avg_tpot,
        "results": results
    }
    
    print(f"\nConcurrent Test Results:")
    print(f"  Concurrent requests: {num_concurrent}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Combined throughput: {concurrent_metrics['combined_throughput']:.1f} tok/s")
    print(f"  Average individual throughput: {avg_throughput:.1f} tok/s")
    print(f"  Average TTFT: {avg_ttft:.3f}s")
    print(f"  Average TPOT: {avg_tpot:.3f}s")
    
    return concurrent_metrics


async def main():
    parser = argparse.ArgumentParser(description="Test LLM streaming generation")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--prompt", default="The future of artificial intelligence is", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--compare", action="store_true", help="Compare streaming vs standard")
    parser.add_argument("--concurrent", type=int, help="Test concurrent streaming requests")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            results = await compare_streaming_vs_standard(args.url, args.prompt, args.max_tokens)
        elif args.concurrent:
            results = await test_concurrent_streaming(args.url, args.concurrent, args.prompt, args.max_tokens)
        else:
            results = await test_streaming(args.url, args.prompt, args.max_tokens)
            
            print("\n" + "=" * 60)
            print("STREAMING TEST RESULTS")
            print("=" * 60)
            print(f"Generated text: '{results['generated_text']}'")
            print(f"Total tokens: {results['total_tokens']}")
            print(f"Total time: {results['total_time']:.3f}s")
            print(f"First token time: {results['first_token_time']:.3f}s")
            print(f"Average throughput: {results['average_throughput']:.1f} tok/s")
            
            if results.get('final_ttft'):
                print(f"Final TTFT: {results['final_ttft']:.3f}s")
                print(f"Final TPOT: {results['final_tpot']:.3f}s")
                print(f"Final throughput: {results['final_throughput']:.1f} tok/s")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
