# Test Suite

Comprehensive test suite for the LLM Latency Bottleneck Analysis project.

## Overview

The test suite includes:
- **Unit tests**: Fast, isolated tests for individual components
- **Integration tests**: Tests requiring actual models or external services
- **API tests**: FastAPI endpoint testing with mocked dependencies

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_api.py
pytest tests/test_inference.py
```

### Run by Category

```bash
# Run only unit tests (fast)
pytest -m unit

# Run only API tests
pytest -m api

# Run only inference tests
pytest -m inference

# Skip slow tests
pytest -m "not slow"

# Skip integration tests
pytest -m "not integration"
```

### Run Specific Test Class or Function

```bash
# Run specific class
pytest tests/test_api.py::TestHealthEndpoint

# Run specific test
pytest tests/test_api.py::TestHealthEndpoint::test_health_check_success
```

### Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

### Show Print Statements

```bash
pytest -s
```

### Stop on First Failure

```bash
pytest -x
```

### Run Last Failed Tests

```bash
pytest --lf
```

## Test Organization

### test_api.py

Tests for FastAPI endpoints:

- **TestHealthEndpoint**: Health check endpoint
- **TestGenerateEndpoint**: Text generation endpoint
  - Request validation
  - Error handling
  - Response structure
- **TestMetricsEndpoint**: Prometheus metrics
- **TestModelEndpoints**: Model management (load/unload/info)
- **TestMiddleware**: Middleware functionality
- **TestCORS**: CORS configuration
- **TestErrorHandling**: Error responses
- **TestIntegration**: End-to-end tests (marked as integration)

### test_inference.py

Tests for inference engine and metrics:

- **TestTokenTimings**: TokenTimings dataclass
  - Average TPOT calculation
  - Total tokens counting
- **TestInferenceTimer**: Performance timing
  - Start/stop workflow
  - Token recording
  - Memory tracking
  - Context manager usage
  - Tracer integration
- **TestInferenceEngine**: Inference engine
  - Initialization
  - Model loading/unloading
  - Generation (mocked)
  - Error handling
- **TestIntegrationInference**: Actual model tests (marked as integration)

## Test Coverage

### Generate Coverage Report

```bash
# Install pytest-cov
pip install pytest-cov

# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# View HTML report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
```

### Coverage Targets

- **Overall**: > 80%
- **API endpoints**: > 90%
- **Inference engine**: > 85%
- **Metrics collection**: > 80%

## Mocking Strategy

### When to Mock

1. **External dependencies**: HuggingFace models, APIs
2. **Expensive operations**: Model loading, inference
3. **Non-deterministic behavior**: Timing, randomness
4. **External services**: Jaeger, Prometheus

### Mock Examples

```python
# Mock inference engine
@patch("src.api.main.inference_engine")
def test_with_mock_engine(mock_engine):
    mock_engine.is_loaded.return_value = True
    mock_engine.generate.return_value = {...}
    # Test code here

# Mock transformers
@patch("src.inference.engine.AutoModelForCausalLM")
@patch("src.inference.engine.AutoTokenizer")
def test_model_loading(mock_model, mock_tokenizer):
    # Test code here
```

## Integration Tests

Integration tests are marked with `@pytest.mark.integration` and typically:
- Download actual models
- Make real API calls
- Require external services

### Running Integration Tests

```bash
# Run all integration tests
pytest -m integration

# Skip integration tests (default for CI)
pytest -m "not integration"
```

### Prerequisites for Integration Tests

1. **Model downloaded**:
   ```bash
   python scripts/download_model.py
   ```

2. **Services running**:
   ```bash
   cd docker
   docker-compose up -d
   ```

3. **Sufficient resources**:
   - 16GB+ RAM
   - GPU (optional but recommended)

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run unit tests
      run: pytest -m "not integration and not slow"
    
    - name: Generate coverage
      run: pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Writing New Tests

### Test Structure

```python
class TestMyFeature:
    """Tests for my feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = "test"
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == expected
    
    @pytest.fixture
    def my_fixture(self):
        """Fixture providing test data."""
        return {"key": "value"}
    
    def test_with_fixture(self, my_fixture):
        """Test using fixture."""
        assert my_fixture["key"] == "value"
```

### Best Practices

1. **Descriptive names**: `test_generate_with_empty_prompt_raises_error`
2. **Arrange-Act-Assert**: Clear test structure
3. **One assertion per test**: Keep tests focused
4. **Use fixtures**: Reuse test setup
5. **Mark appropriately**: Use markers (unit, integration, slow)
6. **Document edge cases**: Test boundaries and errors
7. **Keep tests independent**: No test should depend on another

### Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Fixtures: Descriptive names (no prefix needed)

## Debugging Tests

### Run with PDB

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start of test
pytest --trace
```

### Print Debug Info

```bash
# Show local variables on failure
pytest -l

# Show captured output
pytest -s
```

### Run Single Test in Debug Mode

```bash
pytest tests/test_api.py::TestHealthEndpoint::test_health_check_success -s -v
```

## Common Issues

### Import Errors

```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use editable install
pip install -e .
```

### Mock Not Working

- Check import paths match exactly
- Use `patch` with full module path
- Verify patch location (where object is used, not where it's defined)

### Async Tests Failing

- Ensure `pytest-asyncio` is installed
- Use `@pytest.mark.asyncio` for async tests
- Check `asyncio_mode = auto` in pytest.ini

### Fixtures Not Found

- Fixtures must be in `conftest.py` or same file
- Check fixture scope (function, class, module, session)
- Verify fixture name matches parameter name

## Test Fixtures

### Common Fixtures

```python
# In conftest.py or test file

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def mock_engine():
    """Mocked inference engine."""
    engine = MagicMock()
    engine.is_loaded.return_value = True
    return engine

@pytest.fixture(scope="session")
def test_model():
    """Load actual test model (slow)."""
    # This runs once per session
    engine = InferenceEngine(...)
    engine.load_model()
    yield engine
    engine.unload_model()
```

## Performance Testing

For load/performance testing, use Locust (not pytest):

```bash
cd load_testing
locust -f locustfile.py --host http://localhost:8000
```

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
