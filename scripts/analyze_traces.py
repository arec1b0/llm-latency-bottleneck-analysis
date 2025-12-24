"""
Trace Analysis Script

Analyzes Jaeger traces to identify performance bottlenecks.
Connects to Jaeger API and generates insights.
"""

import json
import logging
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

import requests
from requests.exceptions import RequestException


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class JaegerAnalyzer:
    """
    Analyzes Jaeger traces to identify bottlenecks.
    
    Connects to Jaeger API and provides insights on:
    - Slowest operations
    - Most frequent errors
    - Latency distribution
    - Bottleneck identification
    """
    
    def __init__(self, jaeger_url: str = "http://localhost:16686"):
        """
        Initialize Jaeger analyzer.
        
        Args:
            jaeger_url: Jaeger UI URL
        """
        self.jaeger_url = jaeger_url
        self.api_url = f"{jaeger_url}/api"
        
    def test_connection(self) -> bool:
        """
        Test connection to Jaeger.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            response = requests.get(f"{self.jaeger_url}", timeout=5)
            return response.status_code == 200
        except RequestException:
            return False
    
    def get_services(self) -> List[str]:
        """
        Get list of services in Jaeger.
        
        Returns:
            List of service names
        """
        try:
            response = requests.get(f"{self.api_url}/services", timeout=10)
            response.raise_for_status()
            return response.json().get("data", [])
        except RequestException as e:
            logger.error(f"Failed to get services: {e}")
            return []
    
    def get_traces(
        self,
        service: str,
        lookback: str = "1h",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get traces for a service.
        
        Args:
            service: Service name
            lookback: Time window (e.g., '1h', '2h', '1d')
            limit: Maximum number of traces
        
        Returns:
            List of trace data
        """
        try:
            params = {
                "service": service,
                "lookback": lookback,
                "limit": limit,
            }
            
            response = requests.get(
                f"{self.api_url}/traces",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except RequestException as e:
            logger.error(f"Failed to get traces: {e}")
            return []
    
    def analyze_traces(
        self,
        service: str = "llm-inference-api",
        lookback: str = "1h",
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze traces and generate insights.
        
        Args:
            service: Service name
            lookback: Time window
            limit: Maximum traces to analyze
        
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Fetching traces for service '{service}'...")
        traces = self.get_traces(service, lookback, limit)
        
        if not traces:
            logger.warning("No traces found")
            return {}
        
        logger.info(f"Analyzing {len(traces)} traces...")
        
        # Collect metrics
        operation_durations = defaultdict(list)
        operation_counts = defaultdict(int)
        span_durations = defaultdict(list)
        errors = []
        
        for trace in traces:
            for span in trace.get("spans", []):
                operation = span.get("operationName", "unknown")
                duration_us = span.get("duration", 0)
                duration_ms = duration_us / 1000.0
                
                operation_durations[operation].append(duration_ms)
                operation_counts[operation] += 1
                
                # Track span-level durations
                process = span.get("processID", "")
                span_name = f"{process}::{operation}"
                span_durations[span_name].append(duration_ms)
                
                # Check for errors
                tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
                if tags.get("error") == "true" or tags.get("error") is True:
                    errors.append({
                        "operation": operation,
                        "trace_id": trace.get("traceID"),
                        "error_message": tags.get("error.message", "Unknown error"),
                    })
        
        # Calculate statistics
        analysis = {
            "summary": {
                "total_traces": len(traces),
                "unique_operations": len(operation_durations),
                "total_errors": len(errors),
                "time_window": lookback,
            },
            "operations": self._analyze_operations(operation_durations, operation_counts),
            "slowest_operations": self._get_slowest_operations(operation_durations),
            "bottlenecks": self._identify_bottlenecks(span_durations),
            "errors": errors[:10],  # Top 10 errors
        }
        
        return analysis
    
    def _analyze_operations(
        self,
        durations: Dict[str, List[float]],
        counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Analyze operation-level statistics."""
        operations = []
        
        for op, dur_list in durations.items():
            if not dur_list:
                continue
            
            sorted_durations = sorted(dur_list)
            operations.append({
                "name": op,
                "count": counts[op],
                "min_ms": min(dur_list),
                "max_ms": max(dur_list),
                "avg_ms": sum(dur_list) / len(dur_list),
                "p50_ms": sorted_durations[len(sorted_durations) // 2],
                "p95_ms": sorted_durations[int(len(sorted_durations) * 0.95)],
                "p99_ms": sorted_durations[int(len(sorted_durations) * 0.99)],
            })
        
        # Sort by average duration
        operations.sort(key=lambda x: x["avg_ms"], reverse=True)
        return operations
    
    def _get_slowest_operations(
        self,
        durations: Dict[str, List[float]],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get slowest operations."""
        slowest = []
        
        for op, dur_list in durations.items():
            if not dur_list:
                continue
            
            max_duration = max(dur_list)
            slowest.append({
                "operation": op,
                "max_duration_ms": max_duration,
                "occurrences": len([d for d in dur_list if d > max_duration * 0.8]),
            })
        
        slowest.sort(key=lambda x: x["max_duration_ms"], reverse=True)
        return slowest[:top_n]
    
    def _identify_bottlenecks(
        self,
        span_durations: Dict[str, List[float]],
    ) -> List[Dict[str, Any]]:
        """Identify potential bottlenecks."""
        bottlenecks = []
        
        for span_name, durations in span_durations.items():
            if not durations or len(durations) < 5:
                continue
            
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            
            # Flag as bottleneck if max > 2x average
            if max_duration > 2 * avg_duration and avg_duration > 100:
                bottlenecks.append({
                    "span": span_name,
                    "avg_duration_ms": avg_duration,
                    "max_duration_ms": max_duration,
                    "ratio": max_duration / avg_duration,
                    "occurrences": len(durations),
                })
        
        bottlenecks.sort(key=lambda x: x["ratio"], reverse=True)
        return bottlenecks[:10]
    
    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print formatted analysis results."""
        if not analysis:
            logger.warning("No analysis data to display")
            return
        
        print("\n" + "=" * 80)
        print("JAEGER TRACE ANALYSIS")
        print("=" * 80)
        
        # Summary
        summary = analysis.get("summary", {})
        print(f"\n📊 Summary:")
        print(f"   Total Traces:       {summary.get('total_traces', 0)}")
        print(f"   Unique Operations:  {summary.get('unique_operations', 0)}")
        print(f"   Total Errors:       {summary.get('total_errors', 0)}")
        print(f"   Time Window:        {summary.get('time_window', 'unknown')}")
        
        # Operations
        operations = analysis.get("operations", [])
        if operations:
            print(f"\n⏱️  Operations Performance:")
            print(f"   {'Operation':<40} {'Count':>8} {'Avg (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10}")
            print(f"   {'-' * 78}")
            
            for op in operations[:10]:
                print(f"   {op['name']:<40} {op['count']:>8} "
                      f"{op['avg_ms']:>10.2f} {op['p95_ms']:>10.2f} {op['p99_ms']:>10.2f}")
        
        # Slowest operations
        slowest = analysis.get("slowest_operations", [])
        if slowest:
            print(f"\n🐌 Slowest Operations:")
            print(f"   {'Operation':<50} {'Max Duration (ms)':>20}")
            print(f"   {'-' * 70}")
            
            for op in slowest[:5]:
                print(f"   {op['operation']:<50} {op['max_duration_ms']:>20.2f}")
        
        # Bottlenecks
        bottlenecks = analysis.get("bottlenecks", [])
        if bottlenecks:
            print(f"\n🔴 Potential Bottlenecks:")
            print(f"   {'Span':<50} {'Avg (ms)':>12} {'Max (ms)':>12} {'Ratio':>8}")
            print(f"   {'-' * 82}")
            
            for bn in bottlenecks[:5]:
                print(f"   {bn['span']:<50} {bn['avg_duration_ms']:>12.2f} "
                      f"{bn['max_duration_ms']:>12.2f} {bn['ratio']:>8.2f}x")
        
        # Errors
        errors = analysis.get("errors", [])
        if errors:
            print(f"\n❌ Recent Errors:")
            for i, error in enumerate(errors[:5], 1):
                print(f"   {i}. {error['operation']}")
                print(f"      Error: {error.get('error_message', 'Unknown')}")
                print(f"      Trace: {error.get('trace_id', 'Unknown')}")
        
        print("\n" + "=" * 80)
        print(f"View full traces at: {self.jaeger_url}")
        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    # Parse arguments
    service = "llm-inference-api"
    lookback = "1h"
    limit = 100
    jaeger_url = "http://localhost:16686"
    
    if len(sys.argv) > 1:
        service = sys.argv[1]
    if len(sys.argv) > 2:
        lookback = sys.argv[2]
    if len(sys.argv) > 3:
        limit = int(sys.argv[3])
    
    # Create analyzer
    analyzer = JaegerAnalyzer(jaeger_url)
    
    # Test connection
    logger.info(f"Connecting to Jaeger at {jaeger_url}...")
    if not analyzer.test_connection():
        logger.error(f"Failed to connect to Jaeger at {jaeger_url}")
        logger.error("Make sure Jaeger is running:")
        logger.error("  cd docker && docker-compose up -d")
        sys.exit(1)
    
    logger.info("✅ Connected to Jaeger")
    
    # Get services
    services = analyzer.get_services()
    logger.info(f"Available services: {', '.join(services)}")
    
    if service not in services:
        logger.warning(f"Service '{service}' not found in Jaeger")
        logger.info("Available services:")
        for svc in services:
            logger.info(f"  - {svc}")
        sys.exit(1)
    
    # Analyze traces
    analysis = analyzer.analyze_traces(service, lookback, limit)
    
    # Print results
    analyzer.print_analysis(analysis)
    
    # Save to file
    try:
        output_file = f"trace_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save analysis: {e}")


if __name__ == "__main__":
    main()
