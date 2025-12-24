# Docker Configuration for LLM Latency Analysis

This directory contains Docker Compose configuration for running observability infrastructure.

## Services

### Jaeger (Distributed Tracing)
- **UI**: http://localhost:16686
- **Collector HTTP**: http://localhost:14268
- **Agent UDP**: localhost:6831

### Prometheus (Metrics)
- **UI**: http://localhost:9090
- **Metrics Endpoint**: http://localhost:9090/metrics

## Quick Start

### Start All Services

```bash
cd docker
docker-compose up -d
```

### Check Service Status

```bash
docker-compose ps
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f jaeger
docker-compose logs -f prometheus
```

### Stop Services

```bash
docker-compose down
```

### Stop and Remove Volumes (Clean Slate)

```bash
docker-compose down -v
```

## Service Details

### Jaeger All-in-One

Includes:
- **Agent**: Receives traces via UDP
- **Collector**: Processes and stores traces
- **Query**: Serves the UI and API
- **UI**: Web interface for viewing traces

**Ports:**
- 16686: Jaeger UI
- 14268: Collector HTTP endpoint
- 14250: Collector gRPC endpoint
- 6831: Agent (Thrift compact) UDP
- 9411: Zipkin compatible endpoint

**Storage:**
- Type: Badger (embedded key-value store)
- Location: Docker volume `jaeger-data`
- Persistence: Enabled

### Prometheus

**Configuration:**
- Scrape interval: 5 seconds for API, 15 seconds for others
- Storage: Docker volume `prometheus-data`
- Targets:
  - Prometheus self-monitoring
  - LLM API metrics (host.docker.internal:8000)
  - Node exporter (optional)

**Ports:**
- 9090: Prometheus UI and API

## Accessing UIs

### Jaeger UI
```
http://localhost:16686
```

**Key Features:**
- Search traces by service, operation, tags
- View trace timeline and spans
- Analyze latency distributions
- Compare traces

### Prometheus UI
```
http://localhost:9090
```

**Key Features:**
- Query metrics with PromQL
- Graph metrics over time
- View targets and their health
- Alert rules (if configured)

## Troubleshooting

### Jaeger Not Starting

```bash
# Check logs
docker-compose logs jaeger

# Verify ports are not in use
netstat -ano | findstr :16686
netstat -ano | findstr :6831
```

### Prometheus Can't Scrape API

```bash
# Test from container
docker-compose exec prometheus wget -O- http://host.docker.internal:8000/metrics

# Verify API is running
curl http://localhost:8000/health
```

### Storage Issues

```bash
# Check volume size
docker volume inspect docker_jaeger-data
docker volume inspect docker_prometheus-data

# Remove and recreate volumes
docker-compose down -v
docker-compose up -d
```

### Performance Issues

**Reduce Jaeger sampling:**
Edit `docker-compose.yml`:
```yaml
environment:
  - SAMPLING_STRATEGIES_FILE=/etc/jaeger/sampling.json
```

**Reduce Prometheus scrape interval:**
Edit `prometheus/prometheus.yml`:
```yaml
scrape_interval: 30s  # Increase from 5s
```

## Data Retention

### Jaeger
- Default: Indefinite (limited by disk space)
- Recommendation: Implement external storage for production

### Prometheus
- Default: 15 days
- Configure in `docker-compose.yml`:
```yaml
command:
  - '--storage.tsdb.retention.time=7d'
```

## Production Considerations

1. **Use external storage** for Jaeger (Elasticsearch, Cassandra)
2. **Enable authentication** for Jaeger and Prometheus UIs
3. **Configure retention policies** based on disk space
4. **Use Thanos or Cortex** for long-term Prometheus storage
5. **Implement alerting** with Alertmanager
6. **Monitor resource usage** of observability stack itself

## Network Configuration

All services are on the `llm-network` bridge network for inter-service communication.

To connect your API to this network:

```bash
docker network connect llm-network <your-api-container>
```

Or in your API docker-compose:

```yaml
networks:
  - llm-network

networks:
  llm-network:
    external: true
```

## Health Checks

Services include health checks that Docker monitors:

```bash
# Check health status
docker-compose ps

# HEALTHY status indicates service is ready
```

## Upgrading

```bash
# Pull latest images
docker-compose pull

# Recreate containers
docker-compose up -d --force-recreate
```

## References

- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
