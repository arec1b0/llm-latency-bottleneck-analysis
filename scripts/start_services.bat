@echo off
REM Start Services Script for Windows
REM Starts Docker containers and API server

echo ================================================================================
echo Starting LLM Latency Bottleneck Analysis Services
echo ================================================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [OK] Docker is running
echo.

REM Navigate to docker directory
cd docker

REM Start Docker containers
echo ================================================================================
echo Starting Observability Stack (Jaeger + Prometheus)
echo ================================================================================
echo.

docker-compose up -d

if errorlevel 1 (
    echo [ERROR] Failed to start Docker containers
    pause
    exit /b 1
)

echo.
echo [OK] Docker containers started
echo.

REM Wait for services to be ready
echo Waiting for services to initialize...
timeout /t 10 /nobreak >nul

REM Check Jaeger
echo Checking Jaeger...
curl -s http://localhost:16686 >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Jaeger UI not yet available at http://localhost:16686
) else (
    echo [OK] Jaeger UI: http://localhost:16686
)

REM Check Prometheus
echo Checking Prometheus...
curl -s http://localhost:9090 >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Prometheus not yet available at http://localhost:9090
) else (
    echo [OK] Prometheus: http://localhost:9090
)

echo.
echo ================================================================================
echo Services Status
echo ================================================================================
docker-compose ps

echo.
echo ================================================================================
echo Next Steps
echo ================================================================================
echo.
echo 1. Start the API server:
echo    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
echo.
echo 2. Access the services:
echo    API Docs:     http://localhost:8000/docs
echo    Jaeger UI:    http://localhost:16686
echo    Prometheus:   http://localhost:9090
echo.
echo 3. Run load tests:
echo    cd load_testing
echo    locust -f locustfile.py --host http://localhost:8000
echo.
echo ================================================================================

cd ..

pause
