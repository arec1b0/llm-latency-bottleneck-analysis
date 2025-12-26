---
trigger: always_on
---
# Windsurf Project Rules: LLM Inference Optimization

**Project Context:**
This is a high-performance MLOps project optimizing LLM inference infrastructure. We follow a strict engineering thesis: "A system is mature only when it can be described by one diagram, withstands high load, and recovers in minutes."

**Author Preferences:**
- Daniil Krizhanovskyi, MLOps Lead, Zurich
- Expertise: Distributed systems, Kubernetes, MLOps pipelines, LLM deployment
- Values: Reproducibility, observability, reversibility (not velocity)

---

## 1. General Engineering Principles

### 1.1 No Optimization Without Measurement
- Before writing code, establish a baseline metric script (`scripts/benchmark.py`).
- All performance changes must be validated against benchmarks: TTFT, Tokens/Sec, VRAM.
- Log all benchmark results in `docs/BENCHMARKS_LOG.md` with timestamp and config.

### 1.2 Atomic Changes, Not Rewrites
- One feature = one commit. Do not combine quantization + batching in a single PR.
- Each change must have a single, verifiable acceptance criterion.
- If you make an unrequested refactor, ask first.

### 1.3 Prefer Configuration Over Code
- If the underlying engine (vLLM, TGI, TorchServe) supports a feature via flags, use flags.
- Example: Enable FA2 via `--enable-flash-attention-2`, not by rewriting kernels.
- If a Pydantic/dataclass config object exists, extend it; do not hardcode.

### 1.4 Reversibility in Minutes
- Every deployed change must have a documented rollback procedure.
- For model weights: immutable artifact registry with version tags.
- For infrastructure: Blue/Green or Canary strategy, never in-place edits.
- Target: Recovery time < 5 minutes.

---

## 2. Artifact-Driven Work

### 2.1 Planning Phase (Before Implementation)
When starting a complex task:
1. Create or update `PLAN.md` in the repository root.
2. Break down work into 12–20 atomic steps (not vague "implement X", but "Add FastAPI endpoint at `src/api/inference.py` line 42, add Pydantic schema for request, add unit test").
3. Each step must have: **Files to touch**, **Commands to run**, **Acceptance criteria**.
4. Do NOT write implementation code until the plan is reviewed and pinned in the repo.

### 2.2 Design Documents (ADRs)
For architectural decisions (e.g., vLLM vs TGI), create an ADR:
- Filename: `docs/ADR-NNN-SHORT-TITLE.md`
- Sections: Context, Decision, Consequences, Alternatives Considered.
- This becomes the single source of truth for "why we did this."

### 2.3 Benchmarking and Observability
- All optimizations must produce a `docs/BENCHMARKS_LOG.md` entry.
- Format:
  ```
  ## Optimization: [Name] — [Date]
  - Before: TTFT=200ms, Throughput=150 tok/s, VRAM=20GB
  - After: TTFT=80ms, Throughput=400 tok/s, VRAM=22GB
  - Delta: +166% throughput, -60% latency, +2GB peak memory
  - Tradeoff Analysis: [What we gained, what we lost, why it's worth it]
  - Rollback Plan: `git revert [SHA]` + restart service
  ```

---

## 3. Code Quality Standards

### 3.1 Python & Type Safety
- All functions must have type hints. Use `from typing import` and PEP 484.
- Use `pyright` or `mypy` for static type checking. Run before commit.
- For async code: Always use `asyncio`, never `threading` for I/O.
- Use `pydantic` for config objects, never raw dicts.

### 3.2 Testing
- For each feature: Write a repro test or benchmark script first (TDD).
- Acceptance criterion: `pytest` passes with `--cov=` threshold (e.g., 80% coverage on changed lines).
- Performance tests: Use `pytest-benchmark` or custom latency/throughput assertions.

### 3.3 Linting & Formatting
- Format: `black` (line length 100).
- Lint: `pylint` + `flake8` (zero tolerance for F-strings and unused imports).
- Let Cascade auto-fix, but review the fixes.

### 3.4 Documentation
- Every public function: docstring with Args, Returns, Raises, Example.
- Complex logic: Inline comments explaining "why", not "what" (code shows the "what").

---

## 4. MLOps & Infrastructure Conventions

### 4.1 Kubernetes & Helm
- All K8s deployments: Helm charts in `k8s/charts/`.
- No hardcoded values; use `values.yaml` + templating.
- Every chart must include: `HPA`, `PDB`, `ServiceMonitor` (for Prometheus).
- Naming: `metadata.namespace: {{ .Values.namespace }}`, no inline hardcodes.

### 4.2 Docker Images
- Dockerfile: Use multi-stage builds (compile stage + runtime stage).
- Model weights: Do NOT bake into image; use volume mounts or init containers.
- Image tags: `[PROJECT]-[FEATURE]:[VERSION]-[GIT-SHA]` (e.g., `llm-inference-fa2:1.0-abc123f`).

### 4.3 Model Versioning
- Every model version is immutable: `model_id`, `revision` (git SHA or timestamp), `tokenizer_version`, `runtime_config`.
- Store in `registry/models.yaml` with schema:
  ```
  models:
    - id: "meta-llama/Llama-2-7b-hf"
      revision: "abc123f"
      quantization: "4bit-gptq"
      tokenizer: "tiktoken-cl100k_base"
      docker_tag: "llm-inference:1.0-abc123f"
      max_tokens: 2048
      rope_scaling: 1.0
  ```

### 4.4 Observability
- All inference endpoints must expose Prometheus metrics:
  - `inference_requests_total` (counter, labels: model, status)
  - `inference_duration_seconds` (histogram, labels: model, operation)
  - `gpu_memory_used_bytes` (gauge)
  - `kv_cache_hits_total`, `cache_misses_total` (if applicable)
- All logs: structured JSON via `python-json-logger`.
- Tracing: OpenTelemetry spans for request flow (if applicable).

---

## 5. Cascade-Specific Rules (for AI Agent Interaction)

### 5.1 Context Management
- Always use `@-mentions` to pin files: `@src/inference/engine.py`, `@k8s/values.yaml`.
- Do NOT ask me to provide context verbally; use `@` to auto-include.
- If context is large (>100 lines), ask me to review the selection or create a doc.

### 5.2 Model Selection Strategy
- **Design Phase:** Use Claude 3.5 Sonnet (or Opus 4.5) for architectural decisions (ADRs, PLAN.md).
- **Implementation Phase:** Use SWE-1.5 for code generation (Batching, FA2, Quantization integration).
- **Review Phase:** Verify against benchmarks, then switch back to Sonnet for safety audit.

### 5.3 Prompt Structure for SWE-1.5
When asking SWE-1.5 to implement a feature, follow this template:
```
**Objective:** [One sentence, what will be different after this step]

**Context:**
- @[Pin relevant files]
- Current behavior: [As seen in logs/tests]
- Desired behavior: [After this change]

**Constraints:**
- Do not touch [Files/Systems] unless explicitly approved
- Performance baseline: [TTFT/Throughput/VRAM from benchmarks]

**Acceptance Criteria:**
1. [Testable assertion, e.g., "Unit test passes"]
2. [Performance target, e.g., "TTFT improves by >10%"]
3. [Safety check, e.g., "No regressions in model quality"]

**Steps:**
1. [Step 1 description]
2. [Step 2 description]
...
N. Update PLAN.md with [x] for this task
```

### 5.4 Error Handling
- If Cascade hits an error (NameError, ImportError, etc.), use `Explain and Fix`.
- If performance degrades, rollback immediately: `git revert [SHA]` + run benchmarks again.
- Do not ask Cascade to "fix it faster"; ask for a slower, safer approach instead.

---

## 6. File & Directory Structure

```
.
├── PLAN.md                          # Master task list
├── docs/
│   ├── ARCHITECTURE.md              # One-diagram system description
│   ├── ADR-001-*.md                 # Architecture Decision Records
│   ├── INFERENCE_CURRENT_STATE.md   # Baseline system snapshot
│   ├── BENCHMARKS_LOG.md            # All performance measurements
│   ├── MIGRATION_EVAL.md            # vLLM vs TGI analysis
│   ├── AUTOSCALING.md               # HPA/KEDA strategy
│   ├── MODEL_VERSIONING.md          # Model registry & rollback
│   └── COST_OPTIMIZATION.md         # Cost model & levers
├── scripts/
│   ├── benchmark.py                 # Load testing script
│   ├── test_e2e.py                  # End-to-end integration tests
│   └── validate_deployment.sh       # Deployment health check
├── src/
│   └── inference/
│       ├── engine.py                # Main inference loop
│       ├── config.py                # Pydantic config models
│       ├── models.py                # Model loaders
│       └── api.py                   # FastAPI/async endpoints
├── k8s/
│   └── charts/
│       └── llm-inference/
│           ├── Chart.yaml
│           ├── values.yaml
│           ├── templates/
│           │   ├── deployment.yaml
│           │   ├── hpa.yaml
│           │   ├── pdb.yaml
│           │   └── servicemonitor.yaml
├── registry/
│   └── models.yaml                  # Model version manifest
├── tests/
│   ├── unit/
│   │   └── test_*.py
│   └── integration/
│       └── test_e2e.py
├── .windsurf/
│   └── rules.md                     # This file
├── .codeiumignore                   # Files for Cascade to ignore
└── pytest.ini, pyproject.toml, etc.
```

---

## 7. Common Workflows

### Workflow A: "I want to add a new optimization (e.g., FA2)"
1. **Design:** Describe the feature in natural language → Cascade (Sonnet) creates ADR + PLAN.md.
2. **Validate Plan:** Review the plan. Ask Cascade to refine if needed.
3. **Implement:** Ask Cascade (SWE-1.5) to execute Step 1, then Step 2, etc.
4. **Benchmark:** After each step, run `python scripts/benchmark.py` and append to BENCHMARKS_LOG.md.
5. **Rollback:** If metrics degrade, `git revert [SHA]` immediately.

### Workflow B: "I suspect a performance regression"
1. Run `python scripts/benchmark.py`.
2. Compare against latest entry in `docs/BENCHMARKS_LOG.md`.
3. If regressed: Check recent commits (`git log --oneline`).
4. Revert the suspect commit and re-run benchmarks.
5. Ask Cascade (Sonnet) to analyze what went wrong (ADR-style investigation).

### Workflow C: "Prepare for production deployment"
1. Check: Does PLAN.md have all steps as `[x]`?
2. Check: Do all K8s manifests have `.spec.resources.requests` and `.spec.resources.limits`?
3. Check: Do all model versions in `registry/models.yaml` have rollback tags?
4. Run e2e tests: `pytest tests/integration/ --benchmark`.
5. Run Canary: Deploy to 10% of traffic, wait 5 min, check SLOs.

---

## 8. Exceptions & Escalations

### When to Break the Rules
- **Security vulnerability:** Fix immediately, document post-incident.
- **Production incident:** Hotfix + rollback plan, refactor later.
- **Customer deadline:** Escalate to principal engineer (me) before proceeding without tests/benchmarks.

### Review Checkpoints
- Any change to inference engine config: Requires `@PLAN.md` review.
- Any Docker image change: Requires `@k8s/Dockerfile` review.
- Any K8s deployment: Requires `@k8s/charts/*/templates/deployment.yaml` review.

---

## 9. Commands Reference

Quick commands Cascade should know:

```
# Benchmarking
python scripts/benchmark.py --model meta-llama/Llama-2-7b-hf --batch-size 1,8,32

# Testing
pytest tests/ --cov=src --cov-fail-under=80 -v

# Linting
black src/ && pylint src/ && mypy src/

# Docker Build
docker build -t llm-inference:dev-latest .

# Helm Validate
helm lint k8s/charts/llm-inference/
helm template llm-inference k8s/charts/llm-inference/ -f k8s/charts/llm-inference/values.yaml

# Git Workflow
git checkout -b feature/[task-name]
# ... make changes ...
git add -A && git commit -m "[task-number] [short desc]; docs: update PLAN.md"
```

---

## 10. Questions for Cascade

Use these anchors in prompts to trigger better responses:

- *"How does this change align with 'works under load, recovers in minutes'?"*
- *"What is the rollback plan for this change?"*
- *"How do we measure success for this optimization?"*
- *"What are the tradeoffs (latency vs throughput vs memory)?"*

---

**Last Updated:** 2025-12-25
**Maintained By:** Daniil Krizhanovskyi
```