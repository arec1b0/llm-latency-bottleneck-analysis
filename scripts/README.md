# Utility Scripts

Collection of utility scripts for managing the LLM inference system.

## Available Scripts

### 1. download_model.py

Downloads and caches the LLM model from HuggingFace.

**Usage:**

```bash
# Download default model (from .env)
python scripts/download_model.py

# Download specific model
python scripts/download_model.py mistralai/Mistral-7B-Instruct-v0.2

# Download tokenizer only
python scripts/download_model.py --tokenizer-only
```

**Features:**
- Downloads model weights and tokenizer
- Validates model integrity
- Shows size and parameter information
- Progress tracking
- Disk space verification

**Requirements:**
- Internet connection
- ~20GB free disk space
- HuggingFace authentication (for gated models)

**Authentication (if needed):**
```bash
huggingface-cli login
```

### 2. start_services.bat

Starts Docker containers and prepares the system (Windows).

**Usage:**

```bash
# Double-click the file or run from command prompt
scripts\start_services.bat
```

**What it does:**
1. Checks Docker is running
2. Starts Jaeger and Prometheus containers
3. Verifies services are accessible
4. Displays next steps

**Services started:**
- Jaeger (http://localhost:16686)
- Prometheus (http://localhost:9090)

**Troubleshooting:**
- Ensure Docker Desktop is running
- Check port availability (16686, 9090, 6831)
- View logs: `docker-compose -f docker/docker-compose.yml logs`

### 3. analyze_traces.py

Analyzes Jaeger traces to identify performance bottlenecks.

**Usage:**

```bash
# Analyze last hour of traces
python scripts/analyze_traces.py

# Analyze specific service
python scripts/analyze_traces.py llm-inference-api 2h

# Analyze with custom limit
python scripts/analyze_traces.py llm-inference-api 1h 200
```

**Parameters:**
- `service`: Service name (default: llm-inference-api)
- `lookback`: Time window (default: 1h)
  - Examples: 1h, 2h, 24h, 1d
- `limit`: Max traces to analyze (default: 100)

**Output:**

The script generates:
1. **Console output** with formatted analysis
2. **JSON file** with detailed results

**Analysis includes:**
- Summary statistics
- Operation performance (min, max, avg, P95, P99)
- Slowest operations
- Potential bottlenecks
- Recent errors

**Example output:**

```
📊 Summary:
   Total Traces:       150
   Unique Operations:  12
   Total Errors:       3
   Time Window:        1h

⏱️  Operations Performance:
   Operation                         Count   Avg (ms)   P95 (ms)   P99 (ms)
   -------------------------------------------------------------------------
   POST /generate                      120      2543.5     4200.0     5100.0
   model_generate                      120      2450.3     4100.0     5000.0
   tokenize_input                      120        45.2       78.5       95.0

🐌 Slowest Operations:
   Operation                                  Max Duration (ms)
   ------------------------------------------------------------------
   POST /generate                                       5450.23
   model_generate                                       5350.12

🔴 Potential Bottlenecks:
   Span                                    Avg (ms)     Max (ms)    Ratio
   ----------------------------------------------------------------------
   inference::model_generate                2450.3      5350.1     2.18x
```

**Prerequisites:**
- Jaeger must be running
- Traces must exist in the time window
- `requests` library installed

## Common Workflows

### Initial Setup

```bash
# 1. Start observability stack
scripts\start_services.bat

# 2. Download model
python scripts/download_model.py

# 3. Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 4. Verify
curl http://localhost:8000/health
```

### After Load Testing

```bash
# 1. Analyze traces
python scripts/analyze_traces.py llm-inference-api 30m 500

# 2. Review results
# Check generated JSON file
# View Jaeger UI: http://localhost:16686
```

### Model Management

```bash
# Check model size
python -c "from pathlib import Path; print(sum(f.stat().st_size for f in Path('./models').rglob('*') if f.is_file()) / 1e9, 'GB')"

# Clear model cache (careful!)
rm -rf ./models/*

# Re-download
python scripts/download_model.py
```

### Troubleshooting

**Model download fails:**
```bash
# Check connection
ping huggingface.co

# Check disk space
df -h  # Linux/Mac
wmic logicaldisk get size,freespace,caption  # Windows

# Try tokenizer only first
python scripts/download_model.py --tokenizer-only
```

**Services won't start:**
```bash
# Check Docker
docker info

# Check port conflicts
netstat -ano | findstr "16686"  # Windows
lsof -i :16686  # Linux/Mac

# View logs
cd docker
docker-compose logs
```

**No traces in Jaeger:**
```bash
# Verify API is instrumented
curl http://localhost:8000/health

# Make test request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "max_tokens": 10}'

# Check Jaeger
# Wait 10-30 seconds for traces to appear
```

## Environment Variables

All scripts respect environment variables from `.env`:

```bash
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
MODEL_CACHE_DIR=./models
OTEL_SERVICE_NAME=llm-inference-api
OTEL_EXPORTER_JAEGER_AGENT_HOST=localhost
OTEL_EXPORTER_JAEGER_AGENT_PORT=6831
```

## Additional Tools

### Quick Health Check

```bash
# Check all services
python -c "
import requests
services = {
    'API': 'http://localhost:8000/health',
    'Jaeger': 'http://localhost:16686',
    'Prometheus': 'http://localhost:9090',
}
for name, url in services.items():
    try:
        r = requests.get(url, timeout=2)
        status = '✅' if r.status_code == 200 else '❌'
    except:
        status = '❌'
    print(f'{status} {name}: {url}')
"
```

### Model Info

```bash
# Quick model info
python -c "
from transformers import AutoConfig
import os
from dotenv import load_dotenv
load_dotenv()
model_name = os.getenv('MODEL_NAME', 'mistralai/Mistral-7B-Instruct-v0.2')
config = AutoConfig.from_pretrained(model_name)
print(f'Model: {model_name}')
print(f'Hidden size: {config.hidden_size}')
print(f'Layers: {config.num_hidden_layers}')
print(f'Vocab: {config.vocab_size:,}')
"
```

### Clear Traces

```bash
# Clear Jaeger data (resets traces)
cd docker
docker-compose down -v
docker-compose up -d
```

## References

- [HuggingFace Models](https://huggingface.co/models)
- [Jaeger Tracing](https://www.jaegertracing.io/)
- [Docker Compose](https://docs.docker.com/compose/)
