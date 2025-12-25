# LLM Inference Platform Migration - Implementation Plan

**Document Version**: 1.0
**Status**: APPROVED
**Author**: Principal MLOps Architect
**Date**: 2025-12-25

---

## Executive Summary

This document provides a rigorous, step-by-step execution plan for migrating from the current custom PyTorch/Transformers inference engine to a production-grade vLLM-based Kubernetes deployment with auto-scaling, model versioning, and cost optimization.

**Target Outcomes**:
- Throughput: 0.12 req/s -> 100+ req/s
- TTFT P95: 7.7s -> < 500ms
- Zero-downtime deployments
- Sub-1-minute rollbacks

---

## Phase 1: vLLM Migration (Goals 1)

### Step 1.1: Create vLLM Dockerfile

**File**: `docker/vllm/Dockerfile`

```dockerfile
# Create production vLLM container with model baked in
FROM vllm/vllm-openai:v0.4.3

# Labels for versioning
ARG MODEL_VERSION=1.0.0
ARG MODEL_SHA=undefined
LABEL model.version="${MODEL_VERSION}"
LABEL model.sha256="${MODEL_SHA}"

# Environment defaults
ENV MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.95

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose OpenAI-compatible API
EXPOSE 8000

# Start vLLM server
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--model", "${MODEL_NAME}", \
     "--max-model-len", "${MAX_MODEL_LEN}", \
     "--gpu-memory-utilization", "${GPU_MEMORY_UTILIZATION}"]
```

**Definition of Done**:
- [ ] `docker build -t llm-inference:vllm-test -f docker/vllm/Dockerfile .` succeeds
- [ ] `docker run --gpus all llm-inference:vllm-test` starts without errors
- [ ] `curl http://localhost:8000/health` returns 200 OK

---

### Step 1.2: Create Kubernetes Base Manifests

**File**: `k8s/base/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
  labels:
    app: llm-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      terminationGracePeriodSeconds: 60
      containers:
        - name: vllm
          image: llm-inference:latest
          ports:
            - containerPort: 8000
              name: http
          env:
            - name: MODEL_NAME
              valueFrom:
                configMapKeyRef:
                  name: llm-config
                  key: model_name
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: 32Gi
            requests:
              nvidia.com/gpu: 1
              memory: 24Gi
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
```

**File**: `k8s/base/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-inference
spec:
  selector:
    app: llm-inference
  ports:
    - port: 80
      targetPort: 8000
      name: http
  type: ClusterIP
```

**File**: `k8s/base/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-config
data:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  max_model_len: "4096"
  gpu_memory_utilization: "0.95"
```

**Definition of Done**:
- [ ] `kubectl apply -f k8s/base/` succeeds
- [ ] `kubectl get pods -l app=llm-inference` shows 1/1 Running
- [ ] `kubectl port-forward svc/llm-inference 8000:80` works
- [ ] `curl http://localhost:8000/v1/models` returns model list

---

### Step 1.3: Create Helm Chart

**File**: `helm/llm-inference/Chart.yaml`

```yaml
apiVersion: v2
name: llm-inference
description: vLLM-based LLM inference platform
version: 1.0.0
appVersion: "0.4.3"
```

**File**: `helm/llm-inference/values.yaml`

```yaml
replicaCount: 1

image:
  repository: llm-inference
  tag: latest
  pullPolicy: IfNotPresent

model:
  name: mistralai/Mistral-7B-Instruct-v0.2
  maxLength: 4096
  gpuMemoryUtilization: 0.95
  quantization: null  # or "awq", "gptq"

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 32Gi
  requests:
    nvidia.com/gpu: 1
    memory: 24Gi

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 10
  targetGPUUtilization: 80

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: false
  className: alb
  hosts:
    - host: llm.example.com
      paths:
        - path: /
          pathType: Prefix

metrics:
  enabled: true
  serviceMonitor:
    enabled: true
```

**Definition of Done**:
- [ ] `helm lint helm/llm-inference` passes
- [ ] `helm install llm-test helm/llm-inference --dry-run` succeeds
- [ ] `helm install llm-test helm/llm-inference` deploys successfully
- [ ] `helm upgrade llm-test helm/llm-inference --set replicaCount=2` scales to 2 pods

---

### Step 1.4: Add Prometheus Metrics Endpoint

**Task**: Verify vLLM exposes metrics at `/metrics`

vLLM natively exposes Prometheus metrics. Configure ServiceMonitor:

**File**: `k8s/monitoring/servicemonitor.yaml`

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: llm-inference
  labels:
    app: llm-inference
spec:
  selector:
    matchLabels:
      app: llm-inference
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

**Definition of Done**:
- [ ] `curl http://localhost:8000/metrics` returns Prometheus format
- [ ] Prometheus scrapes `llm-inference` target successfully
- [ ] `vllm:num_requests_running` metric visible in Prometheus
- [ ] `vllm:avg_generation_throughput_toks_per_s` metric visible

---

### Step 1.5: Validate OpenAI API Compatibility

**Task**: Create compatibility test suite

**File**: `tests/integration/test_openai_compat.py`

```python
import openai
import pytest

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

def test_completions_endpoint():
    response = client.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        prompt="Hello, my name is",
        max_tokens=50
    )
    assert response.choices[0].text
    assert response.usage.completion_tokens > 0

def test_chat_completions_endpoint():
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=50
    )
    assert response.choices[0].message.content

def test_streaming():
    stream = client.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        prompt="Count to 5:",
        max_tokens=50,
        stream=True
    )
    tokens = list(stream)
    assert len(tokens) > 0
```

**Definition of Done**:
- [ ] `pytest tests/integration/test_openai_compat.py` passes (all 3 tests)
- [ ] Streaming responses work correctly
- [ ] Response format matches OpenAI API spec

---

## Phase 2: Auto-Scaling Implementation (Goal 2)

### Step 2.1: Install KEDA

**Task**: Deploy KEDA to cluster

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
```

**Definition of Done**:
- [ ] `kubectl get pods -n keda` shows keda-operator Running
- [ ] `kubectl get crd | grep scaledobjects` returns scaledobjects.keda.sh

---

### Step 2.2: Create KEDA ScaledObject

**File**: `k8s/scaling/scaledobject.yaml`

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-inference-scaler
spec:
  scaleTargetRef:
    name: llm-inference
  pollingInterval: 15
  cooldownPeriod: 300
  minReplicaCount: 1
  maxReplicaCount: 10
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
            - type: Pods
              value: 1
              periodSeconds: 60
        scaleUp:
          stabilizationWindowSeconds: 0
          policies:
            - type: Pods
              value: 2
              periodSeconds: 30
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: vllm_queue_depth
        query: sum(vllm:num_requests_waiting{service="llm-inference"})
        threshold: "10"
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: vllm_gpu_utilization
        query: avg(vllm:gpu_cache_usage_perc{service="llm-inference"})
        threshold: "85"
```

**Definition of Done**:
- [ ] `kubectl apply -f k8s/scaling/scaledobject.yaml` succeeds
- [ ] `kubectl get scaledobject llm-inference-scaler` shows Ready
- [ ] HPA is automatically created: `kubectl get hpa`

---

### Step 2.3: Implement Graceful Shutdown Handler

**Task**: Ensure active requests complete before pod termination

**File**: `k8s/base/deployment.yaml` (add to pod spec)

```yaml
spec:
  terminationGracePeriodSeconds: 120
  containers:
    - name: vllm
      lifecycle:
        preStop:
          exec:
            command:
              - /bin/sh
              - -c
              - |
                # Wait for active requests to complete
                while [ $(curl -s localhost:8000/metrics | grep 'vllm:num_requests_running' | awk '{print $2}') -gt 0 ]; do
                  echo "Waiting for active requests to complete..."
                  sleep 5
                done
                echo "All requests completed, shutting down"
```

**File**: `k8s/base/pdb.yaml`

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: llm-inference-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: llm-inference
```

**Definition of Done**:
- [ ] `kubectl apply -f k8s/base/pdb.yaml` succeeds
- [ ] During pod termination, active requests complete (verified via logs)
- [ ] `kubectl delete pod llm-inference-xxx` waits for graceful shutdown
- [ ] No 502/503 errors during rolling update

---

### Step 2.4: Load Test Scaling Behavior

**File**: `tests/load/test_autoscaling.py`

```python
"""
Load test to verify auto-scaling behavior.
Run with: locust -f tests/load/test_autoscaling.py --host http://llm-service
"""
from locust import HttpUser, task, between

class LLMUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def generate(self):
        self.client.post("/v1/completions", json={
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "prompt": "Explain quantum computing in one sentence.",
            "max_tokens": 50
        })
```

**Test Procedure**:
1. Start with 1 replica
2. Ramp to 50 concurrent users over 5 minutes
3. Observe scaling up (expect 3-5 replicas)
4. Reduce to 5 users
5. Observe scaling down after 5-minute cooldown

**Definition of Done**:
- [ ] Pods scale up when queue_depth > 10 for > 30s
- [ ] Pods scale up when GPU utilization > 85% for > 30s
- [ ] Pods scale down only after 5-minute cooldown
- [ ] No requests dropped during scaling events
- [ ] Zero 502/503 errors during scale-down

---

## Phase 3: Model Versioning (Goal 3)

### Step 3.1: Create S3 Model Registry

**File**: `terraform/model-registry/main.tf`

```hcl
resource "aws_s3_bucket" "model_registry" {
  bucket = "llm-model-registry-${var.environment}"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_dynamodb_table" "model_versions" {
  name         = "llm-model-versions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "model_name"
  range_key    = "version"

  attribute {
    name = "model_name"
    type = "S"
  }

  attribute {
    name = "version"
    type = "S"
  }
}
```

**Definition of Done**:
- [ ] `terraform apply` creates S3 bucket and DynamoDB table
- [ ] `aws s3 ls s3://llm-model-registry-prod/` works
- [ ] `aws dynamodb scan --table-name llm-model-versions` works

---

### Step 3.2: Create Model Upload Script

**File**: `scripts/upload_model.py`

```python
#!/usr/bin/env python3
"""
Upload model to S3 registry with version tracking.

Usage:
    python scripts/upload_model.py \
        --model-path ./models/mistral-7b \
        --version 1.0.0 \
        --model-name mistral-7b-instruct
"""
import argparse
import hashlib
import json
import boto3
from pathlib import Path
from datetime import datetime

def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:12]

def upload_model(model_path: Path, version: str, model_name: str):
    s3 = boto3.client("s3")
    dynamodb = boto3.resource("dynamodb")
    bucket = "llm-model-registry-prod"
    table = dynamodb.Table("llm-model-versions")

    # Compute hashes
    model_sha = compute_sha256(model_path / "model.safetensors")
    tokenizer_sha = compute_sha256(model_path / "tokenizer.json")
    config_sha = compute_sha256(model_path / "config.json")

    # Create manifest
    manifest = {
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "model": {"sha256": model_sha},
        "tokenizer": {"sha256": tokenizer_sha},
        "config": {"sha256": config_sha},
        "docker_image": f"llm-inference:v{version}-{model_sha}",
        "rollback_safe": True
    }

    # Upload to S3
    prefix = f"models/{model_name}/v{version}/"
    for file in model_path.iterdir():
        s3.upload_file(str(file), bucket, f"{prefix}{file.name}")

    # Upload manifest
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}MANIFEST.json",
        Body=json.dumps(manifest, indent=2)
    )

    # Update DynamoDB
    table.put_item(Item={
        "model_name": model_name,
        "version": version,
        **manifest
    })

    print(f"Uploaded {model_name} v{version}")
    print(f"Docker image tag: {manifest['docker_image']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()
    upload_model(Path(args.model_path), args.version, args.model_name)
```

**Definition of Done**:
- [ ] `python scripts/upload_model.py --model-path ./models/test --version 1.0.0 --model-name test` succeeds
- [ ] Files visible in S3: `aws s3 ls s3://llm-model-registry-prod/models/test/v1.0.0/`
- [ ] MANIFEST.json contains correct SHAs
- [ ] DynamoDB entry created with version metadata

---

### Step 3.3: Implement Rollback Script

**File**: `scripts/rollback.sh`

```bash
#!/bin/bash
set -e

# Usage: ./scripts/rollback.sh <version>
# Example: ./scripts/rollback.sh 1.0.0

VERSION=${1:?"Usage: $0 <version>"}
MODEL_NAME=${MODEL_NAME:-"mistral-7b-instruct"}
NAMESPACE=${NAMESPACE:-"production"}

echo "Rolling back to version ${VERSION}..."

# 1. Get image tag from DynamoDB
IMAGE_TAG=$(aws dynamodb get-item \
    --table-name llm-model-versions \
    --key '{"model_name": {"S": "'${MODEL_NAME}'"}, "version": {"S": "'${VERSION}'"}}' \
    --query 'Item.docker_image.S' \
    --output text)

if [ "$IMAGE_TAG" == "None" ]; then
    echo "ERROR: Version ${VERSION} not found in registry"
    exit 1
fi

echo "Target image: ${IMAGE_TAG}"

# 2. Update deployment
START_TIME=$(date +%s)

kubectl set image deployment/llm-inference \
    vllm=${IMAGE_TAG} \
    -n ${NAMESPACE}

# 3. Wait for rollout
kubectl rollout status deployment/llm-inference \
    -n ${NAMESPACE} \
    --timeout=120s

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Rollback completed in ${DURATION} seconds"

# 4. Verify health
sleep 10
HEALTH=$(kubectl exec -n ${NAMESPACE} \
    $(kubectl get pod -n ${NAMESPACE} -l app=llm-inference -o jsonpath='{.items[0].metadata.name}') \
    -- curl -s localhost:8000/health | jq -r '.status')

if [ "$HEALTH" == "ok" ]; then
    echo "Rollback successful! Service is healthy."
else
    echo "WARNING: Health check returned: ${HEALTH}"
    exit 1
fi
```

**Definition of Done**:
- [ ] `./scripts/rollback.sh 1.0.0` completes in < 60 seconds
- [ ] Deployment image updated to correct tag
- [ ] Health check passes after rollback
- [ ] No 5xx errors during rollback (verified via Prometheus)

---

### Step 3.4: Create CI/CD Pipeline for Model Promotion

**File**: `.github/workflows/model-release.yaml`

```yaml
name: Model Release

on:
  push:
    tags:
      - 'model-v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Extract version
        id: version
        run: echo "VERSION=${GITHUB_REF#refs/tags/model-v}" >> $GITHUB_OUTPUT

      - name: Build Docker image
        run: |
          docker build \
            --build-arg MODEL_VERSION=${{ steps.version.outputs.VERSION }} \
            -t llm-inference:v${{ steps.version.outputs.VERSION }} \
            -f docker/vllm/Dockerfile .

      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker push $ECR_REGISTRY/llm-inference:v${{ steps.version.outputs.VERSION }}

      - name: Update deployment (staging)
        run: |
          kubectl set image deployment/llm-inference \
            vllm=$ECR_REGISTRY/llm-inference:v${{ steps.version.outputs.VERSION }} \
            -n staging

      - name: Run smoke tests
        run: |
          sleep 60  # Wait for rollout
          pytest tests/integration/test_openai_compat.py

      - name: Promote to production
        if: success()
        run: |
          kubectl set image deployment/llm-inference \
            vllm=$ECR_REGISTRY/llm-inference:v${{ steps.version.outputs.VERSION }} \
            -n production
```

**Definition of Done**:
- [ ] `git tag model-v1.1.0 && git push --tags` triggers pipeline
- [ ] Docker image built and pushed to ECR
- [ ] Staging deployment updated automatically
- [ ] Smoke tests pass before production promotion
- [ ] Production deployment updated after successful tests

---

## Phase 4: Cost Optimization (Goal 4)

### Step 4.1: Enable AWQ Quantization

**Task**: Configure vLLM with AWQ quantization

**File**: `helm/llm-inference/values-prod.yaml`

```yaml
model:
  name: TheBloke/Mistral-7B-Instruct-v0.2-AWQ
  quantization: awq
  gpuMemoryUtilization: 0.95

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 16Gi  # Reduced from 32Gi due to quantization
  requests:
    nvidia.com/gpu: 1
    memory: 12Gi
```

**Definition of Done**:
- [ ] `helm upgrade llm-prod helm/llm-inference -f helm/llm-inference/values-prod.yaml` succeeds
- [ ] Model loads successfully with AWQ
- [ ] GPU memory usage < 12GB (vs 24GB unquantized)
- [ ] Quality validation: BLEU score within 2% of FP16 baseline

---

### Step 4.2: Configure Spot Instances

**File**: `k8s/base/deployment.yaml` (add node affinity)

```yaml
spec:
  template:
    spec:
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              preference:
                matchExpressions:
                  - key: node.kubernetes.io/lifecycle
                    operator: In
                    values:
                      - spot
      tolerations:
        - key: "spot"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
```

**File**: `terraform/eks-nodegroup.tf`

```hcl
resource "aws_eks_node_group" "gpu_spot" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "gpu-spot"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.private[*].id
  capacity_type   = "SPOT"
  instance_types  = ["g5.xlarge", "g5.2xlarge", "p3.2xlarge"]

  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 1
  }

  labels = {
    "node.kubernetes.io/lifecycle" = "spot"
    "nvidia.com/gpu"               = "true"
  }

  taint {
    key    = "spot"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
}
```

**Definition of Done**:
- [ ] Spot node group created successfully
- [ ] Pods scheduled on spot instances
- [ ] Cost reduction visible in AWS Cost Explorer (expect ~60% savings)
- [ ] Graceful handling of spot interruptions (2-minute warning)

---

### Step 4.3: Implement Semantic Cache

**File**: `k8s/caching/redis.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-cache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-cache
  template:
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          command: ["redis-server", "--maxmemory", "2gb", "--maxmemory-policy", "allkeys-lru"]
          ports:
            - containerPort: 6379
          resources:
            limits:
              memory: 2Gi
            requests:
              memory: 1Gi
```

**File**: `src/cache/semantic_cache.py`

```python
"""
Semantic cache for LLM responses.
Caches responses for semantically similar prompts.
"""
import redis
import hashlib
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, redis_url: str, similarity_threshold: float = 0.95):
        self.redis = redis.from_url(redis_url)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = similarity_threshold

    def get(self, prompt: str) -> str | None:
        """Check cache for similar prompt."""
        embedding = self.encoder.encode(prompt)
        cache_key = self._hash_embedding(embedding)
        return self.redis.get(cache_key)

    def set(self, prompt: str, response: str, ttl: int = 3600):
        """Cache response with TTL."""
        embedding = self.encoder.encode(prompt)
        cache_key = self._hash_embedding(embedding)
        self.redis.setex(cache_key, ttl, response)

    def _hash_embedding(self, embedding) -> str:
        # Quantize embedding to reduce key space
        quantized = (embedding * 100).astype(int)
        return hashlib.md5(quantized.tobytes()).hexdigest()
```

**Definition of Done**:
- [ ] Redis deployed and accessible
- [ ] Cache hit rate > 10% for repeated similar prompts
- [ ] Cache lookup latency < 5ms
- [ ] No stale responses (TTL working correctly)

---

### Step 4.4: Create Cost Dashboard

**File**: `grafana/dashboards/cost.json`

Key panels:
1. **Cost per 1M Tokens** (calculated from usage and pricing)
2. **Spot vs On-Demand Usage** (percentage breakdown)
3. **Cache Hit Rate** (semantic cache effectiveness)
4. **GPU Utilization vs Cost** (efficiency metric)

**PromQL Queries**:

```promql
# Cost per 1M tokens (assuming $1/hour GPU)
(1 / (sum(rate(vllm:prompt_tokens_total[1h])) + sum(rate(vllm:generation_tokens_total[1h])))) * 1000000

# Effective throughput (tokens/dollar)
(sum(rate(vllm:generation_tokens_total[1h]))) / (count(kube_pod_info{pod=~"llm-inference.*"}) * 1)

# Cache hit rate
sum(rate(llm_cache_hits_total[5m])) / sum(rate(llm_cache_requests_total[5m]))
```

**Definition of Done**:
- [ ] Dashboard accessible at Grafana URL
- [ ] All panels rendering data
- [ ] Cost per 1M tokens calculation validated
- [ ] Alert configured for cost > $1/M tokens

---

## Verification Checklist

### Phase 1: vLLM Migration
- [ ] 1.1: Dockerfile builds successfully
- [ ] 1.2: K8s manifests deploy successfully
- [ ] 1.3: Helm chart installs and upgrades work
- [ ] 1.4: Prometheus scrapes vLLM metrics
- [ ] 1.5: OpenAI API compatibility tests pass

### Phase 2: Auto-Scaling
- [ ] 2.1: KEDA installed and running
- [ ] 2.2: ScaledObject triggers scaling
- [ ] 2.3: Graceful shutdown completes all requests
- [ ] 2.4: Load test shows proper scaling behavior

### Phase 3: Model Versioning
- [ ] 3.1: S3 bucket and DynamoDB table created
- [ ] 3.2: Model upload script works
- [ ] 3.3: Rollback completes in < 60 seconds
- [ ] 3.4: CI/CD pipeline promotes models

### Phase 4: Cost Optimization
- [ ] 4.1: AWQ quantization reduces memory by 40%
- [ ] 4.2: Spot instances reduce cost by 60%
- [ ] 4.3: Semantic cache achieves > 10% hit rate
- [ ] 4.4: Cost dashboard shows < $0.60 per 1M tokens

---

## Success Criteria Summary

| Metric | Current | Target | Verification |
|--------|---------|--------|--------------|
| TTFT P95 | 7.7s | < 500ms | Prometheus query |
| TPOT P50 | 300ms | < 50ms | Prometheus query |
| Throughput | 0.12/s | > 10/s | Load test |
| Rollback Time | N/A | < 60s | Timed test |
| Cost per 1M tokens | N/A | < $0.60 | Cost dashboard |
| Zero-downtime deploys | No | Yes | Rolling update test |

---

## Appendix: File Structure

```
.
   docker/
      vllm/
          Dockerfile                 # Step 1.1
   k8s/
      base/
         deployment.yaml           # Step 1.2
         service.yaml              # Step 1.2
         configmap.yaml            # Step 1.2
         pdb.yaml                  # Step 2.3
      scaling/
         scaledobject.yaml         # Step 2.2
      monitoring/
         servicemonitor.yaml       # Step 1.4
      caching/
          redis.yaml                 # Step 4.3
   helm/
      llm-inference/
          Chart.yaml                 # Step 1.3
          values.yaml               # Step 1.3
          values-prod.yaml          # Step 4.1
   terraform/
      model-registry/
         main.tf                    # Step 3.1
      eks-nodegroup.tf              # Step 4.2
   scripts/
      upload_model.py               # Step 3.2
      rollback.sh                   # Step 3.3
   tests/
      integration/
         test_openai_compat.py     # Step 1.5
      load/
          test_autoscaling.py       # Step 2.4
   src/
      cache/
          semantic_cache.py         # Step 4.3
   grafana/
      dashboards/
          cost.json                  # Step 4.4
   .github/
       workflows/
           model-release.yaml         # Step 3.4
```

---

**Document Control**
- Created: 2025-12-25
- Last Updated: 2025-12-25
- Version: 1.0
