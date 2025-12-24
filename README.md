# LLM Latency Bottleneck Analysis

Production-grade project for analyzing latency bottlenecks in LLM inference pipelines using distributed tracing and observability.

## Project Overview

This project deploys a lightweight LLM (Mistral-7B) behind FastAPI with comprehensive OpenTelemetry instrumentation to identify performance bottlenecks in production-like scenarios.

### Key Metrics Tracked

- **TTFT (Time to First Token)**: Latency until the first token is generated
- **TPOT (Time Per Output Token)**: Average time per subsequent token
- **Memory Bandwidth Utilization**: RAM and VRAM usage patterns
- **Request Throughput**: Concurrent request handling capacity
- **Queue Time**: Time spent waiting in the request queue

## Architecture

```
Client → FastAPI → Inference Engine → LLM Model
   ↓         ↓            ↓              ↓
OpenTelemetry Traces → Jaeger UI
```

## Prerequisites

- Python 3.10+
- Docker Desktop (for Jaeger)
- CUDA-capable GPU (recommended) or CPU fallback
- 16GB RAM minimum (32GB recommended)
- 20GB disk space for model weights

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd llm-latency-bottleneck-analysis
```

### 2. Create Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
copy .env.example .env
# Edit .env with your configuration
```

### 4. Start Jaeger

```bash
cd docker
docker-compose up -d
```

### 5. Download Model

```bash
python scripts/download_model.py
```

### 6. Start API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Run Load Tests

```bash
cd load_testing
locust -f locustfile.py --host http://localhost:8000
```

## Usage

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Generate Text
```bash
POST /generate
Content-Type: application/json

{
  "prompt": "Explain quantum computing in simple terms",
  "max_tokens": 256,
  "temperature": 0.7
}
```

#### Metrics
```bash
GET /metrics
```

### Viewing Traces

Open Jaeger UI:
```
http://localhost:16686
```

Filter by service: `llm-inference-api`

## Load Testing Scenarios

### Scenario 1: Baseline (10 concurrent users)
```bash
locust -f locustfile.py --users 10 --spawn-rate 2 --run-time 5m --host http://localhost:8000
```

### Scenario 2: Medium Load (25 concurrent users)
```bash
locust -f locustfile.py --users 25 --spawn-rate 5 --run-time 5m --host http://localhost:8000
```

### Scenario 3: High Load (50 concurrent users)
```bash
locust -f locustfile.py --users 50 --spawn-rate 10 --run-time 5m --host http://localhost:8000
```

## Bottleneck Analysis

Common bottlenecks identified:

1. **Model Loading**: Cold start penalty (first request)
2. **Tokenization**: CPU-bound preprocessing
3. **Memory Transfer**: CPU ↔ GPU data movement
4. **Batch Processing**: Queue management inefficiency
5. **Network I/O**: Serialization overhead

## Project Structure

```
llm-latency-bottleneck-analysis/
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── inference/         # LLM inference engine
│   └── telemetry/         # OpenTelemetry instrumentation
├── tests/                 # Unit and integration tests
├── load_testing/          # Locust load test scenarios
├── scripts/               # Utility scripts
├── docker/                # Docker configurations
└── docs/                  # Documentation
```

## Metrics Interpretation

### TTFT (Time to First Token)
- **Good**: < 500ms
- **Acceptable**: 500ms - 1s
- **Poor**: > 1s

### TPOT (Time Per Output Token)
- **Good**: < 50ms
- **Acceptable**: 50ms - 100ms
- **Poor**: > 100ms

### Memory Bandwidth
- Monitor for OOM errors
- Track VRAM utilization (should stay < 90%)

## Troubleshooting

### Out of Memory
```bash
# Enable 8-bit quantization in .env
MODEL_LOAD_IN_8BIT=true
```

### Slow Inference
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reduce max tokens
MAX_NEW_TOKENS=128
```

### Jaeger Not Accessible
```bash
docker-compose -f docker/docker-compose.yml logs jaeger
```

## Contributing

1. Follow SOLID principles
2. Write tests for new features
3. Update documentation
4. Use meaningful commit messages

## License

MIT License

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Jaeger Tracing](https://www.jaegertracing.io/)
- [Locust Load Testing](https://locust.io/)
