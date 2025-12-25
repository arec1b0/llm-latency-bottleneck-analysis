# Production Monitoring Setup

This directory contains production monitoring configurations for the LLM Inference Service, including alerting rules, Alertmanager configuration, and monitoring best practices.

## Components

### 1. Prometheus Alerting Rules (`production-alerts.yml`)

Comprehensive alerting rules covering:
- **Service Health**: Service availability, model loading status
- **Performance**: TTFT, throughput, error rates
- **Resources**: GPU/CPU/memory utilization
- **Scaling**: Active requests, queue depth, latency
- **Infrastructure**: Monitoring services, disk space

### 2. Alertmanager Configuration (`alertmanager.yml`)

Alert routing and notification configuration:
- **Email notifications** for all alert levels
- **Slack integration** for team channels
- **PagerDuty integration** for critical alerts
- **Alert grouping** and inhibition rules
- **Escalation policies** based on severity

## Setup Instructions

### Prerequisites

1. **Prometheus** with alerting enabled
2. **Alertmanager** service running
3. **Notification channels** configured:
   - Email (SMTP)
   - Slack (webhook URL)
   - PagerDuty (service keys)

### Configuration Steps

#### 1. Configure Prometheus Alerting

Add alerting rules to Prometheus configuration:

```yaml
# prometheus.yml
rule_files:
  - '/etc/prometheus/alerts/*.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'alertmanager:9093'
```

Copy alert rules:
```bash
cp monitoring/production-alerts.yml /etc/prometheus/alerts/
```

#### 2. Configure Alertmanager

Set environment variables for notification channels:

```bash
# Email configuration
export SMTP_PASSWORD="your-smtp-password"

# Slack configuration
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# PagerDuty configuration
export PAGERDUTY_SERVICE_KEY="your-pagerduty-service-key"
export PAGERDUTY_LLM_SERVICE_KEY="your-llm-specific-key"
```

Update email addresses and channels in `alertmanager.yml`:
- Replace `team@example.com` with your team email
- Replace `#alerts-critical` with your Slack channel
- Update PagerDuty service keys

Deploy Alertmanager configuration:
```bash
cp monitoring/alertmanager.yml /etc/alertmanager/
docker-compose restart alertmanager
```

#### 3. Add Alertmanager to Docker Compose

Add to your `docker-compose.yml`:

```yaml
alertmanager:
  image: prom/alertmanager:latest
  container_name: alertmanager
  ports:
    - "9093:9093"
  volumes:
    - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
    - alertmanager-data:/alertmanager
  command:
    - '--config.file=/etc/alertmanager/alertmanager.yml'
    - '--storage.path=/alertmanager'
  networks:
    - llm-network
  restart: unless-stopped
```

#### 4. Update Prometheus Configuration

Ensure Prometheus is configured to send alerts to Alertmanager:

```yaml
# docker/prometheus/prometheus.yml
rule_files:
  - '/etc/prometheus/alerts/production-alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'alertmanager:9093'
```

## Alert Severity Levels

### Critical
- **Response time**: Immediate (< 5 minutes)
- **Notification**: Email, Slack, PagerDuty
- **Examples**: Service down, very high error rate, critical resource exhaustion

### Warning
- **Response time**: Within business hours
- **Notification**: Email, Slack
- **Examples**: High latency, elevated error rate, resource pressure

## Alert Descriptions

### Service Health Alerts

| Alert | Threshold | Duration | Description |
|-------|-----------|----------|-------------|
| LLMServiceDown | Service unavailable | 1m | API service is not responding |
| LLMServiceUnhealthy | Health check failing | 2m | Health endpoint returning errors |
| ModelNotLoaded | Model not loaded | 5m | Inference engine has no model loaded |

### Performance Alerts

| Alert | Threshold | Duration | Description |
|-------|-----------|----------|-------------|
| HighTTFT | TTFT > 2s | 5m | Time to first token is elevated |
| VeryHighTTFT | TTFT > 5s | 2m | Time to first token is critically high |
| LowThroughput | < 10 tok/s | 10m | Token generation rate is low |
| VeryLowThroughput | < 5 tok/s | 5m | Token generation rate is critically low |
| HighErrorRate | > 5% | 3m | Error rate is elevated |
| VeryHighErrorRate | > 10% | 1m | Error rate is critically high |

### Resource Alerts

| Alert | Threshold | Duration | Description |
|-------|-----------|----------|-------------|
| HighGPUUtilization | > 90% | 10m | GPU utilization sustained above 90% |
| HighGPUMemoryUsage | > 85% | 5m | GPU memory usage above 85% |
| CriticalGPUMemoryUsage | > 95% | 2m | GPU memory critically high |
| HighCPUUsage | > 80% | 10m | CPU usage sustained above 80% |
| HighMemoryUsage | > 85% | 10m | System memory usage above 85% |

### Scaling Alerts

| Alert | Threshold | Duration | Description |
|-------|-----------|----------|-------------|
| HighActiveRequests | > 10 | 5m | High number of concurrent requests |
| VeryHighActiveRequests | > 20 | 2m | Very high concurrent requests, may be overloaded |
| RequestQueueBuilding | > 5 | 3m | Request queue is building up |
| HighRequestLatency | p95 > 10s | 5m | 95th percentile latency is high |

## Runbooks

### LLMServiceDown

**Symptoms**: Service health check failing, no responses from API

**Investigation**:
1. Check service status: `docker-compose ps`
2. Check logs: `docker-compose logs llm-inference`
3. Verify model loading: Check GPU memory usage
4. Check resource availability: `nvidia-smi`

**Resolution**:
1. Restart service: `docker-compose restart llm-inference`
2. If OOM error: Reduce batch size or enable 8-bit quantization
3. If model loading fails: Check model cache and download
4. Escalate if issue persists

### HighTTFT / LowThroughput

**Symptoms**: Slow token generation, high latency

**Investigation**:
1. Check GPU utilization: `nvidia-smi`
2. Check concurrent requests: Grafana dashboard
3. Review recent changes: Model, batch size, concurrency
4. Check system resources: CPU, memory

**Resolution**:
1. If GPU underutilized: Increase batch size
2. If GPU maxed out: Reduce concurrent requests
3. If CPU bottleneck: Optimize preprocessing
4. Consider scaling horizontally

### CriticalGPUMemoryUsage

**Symptoms**: GPU memory near capacity, potential OOM

**Investigation**:
1. Check current memory usage: `nvidia-smi`
2. Review batch sizes and concurrent requests
3. Check for memory leaks in logs
4. Verify model quantization settings

**Resolution**:
1. Reduce batch size immediately
2. Enable 8-bit quantization if not already
3. Reduce max concurrent requests
4. Clear GPU cache if needed
5. Consider larger GPU instance

## Testing Alerts

### Test Alert Firing

```bash
# Trigger a test alert
curl -X POST http://localhost:9093/api/v1/alerts -d '[{
  "labels": {
    "alertname": "TestAlert",
    "severity": "warning",
    "service": "llm-inference"
  },
  "annotations": {
    "summary": "Test alert",
    "description": "This is a test alert"
  }
}]'
```

### Verify Alert Routing

1. Check Alertmanager UI: `http://localhost:9093`
2. Verify alerts appear in configured channels
3. Test inhibition rules by firing related alerts
4. Confirm escalation to PagerDuty for critical alerts

## Monitoring Best Practices

### 1. Alert Fatigue Prevention
- Set appropriate thresholds to avoid false positives
- Use inhibition rules to suppress redundant alerts
- Regularly review and tune alert thresholds
- Group related alerts together

### 2. Escalation Policies
- Warning alerts: Team notification during business hours
- Critical alerts: Immediate notification with escalation
- Use PagerDuty for on-call rotation
- Document escalation procedures

### 3. Runbook Maintenance
- Keep runbooks up-to-date with current procedures
- Include recent incident learnings
- Test runbooks regularly
- Link runbooks in alert annotations

### 4. Dashboard Monitoring
- Create dedicated dashboards for each alert type
- Include relevant metrics for troubleshooting
- Set up dashboard snapshots for incidents
- Review dashboards during postmortems

### 5. Alert Review
- Weekly review of alert frequency and accuracy
- Monthly tuning of thresholds based on trends
- Quarterly review of alert coverage
- Annual review of escalation policies

## Metrics to Monitor

### Key Performance Indicators (KPIs)
- **Availability**: Service uptime percentage
- **Latency**: p50, p95, p99 TTFT and total latency
- **Throughput**: Requests per second, tokens per second
- **Error Rate**: Percentage of failed requests
- **Resource Utilization**: GPU, CPU, memory usage

### Business Metrics
- **User Experience**: Time to first token distribution
- **Capacity**: Concurrent request handling
- **Cost Efficiency**: Tokens per dollar, requests per instance
- **Reliability**: Error rate trends, incident frequency

## Integration with CI/CD

### Pre-deployment Checks
```bash
# Validate alert rules
promtool check rules monitoring/production-alerts.yml

# Validate Alertmanager config
amtool check-config monitoring/alertmanager.yml
```

### Post-deployment Verification
```bash
# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify alert rules loaded
curl http://localhost:9090/api/v1/rules

# Check Alertmanager status
curl http://localhost:9093/api/v1/status
```

## Troubleshooting

### Alerts Not Firing

1. Check Prometheus rule evaluation:
   ```bash
   curl http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.name=="YourAlertName")'
   ```

2. Verify metrics are being collected:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=llm_requests_total'
   ```

3. Check Prometheus logs:
   ```bash
   docker-compose logs prometheus
   ```

### Alerts Not Routing

1. Check Alertmanager configuration:
   ```bash
   amtool config show
   ```

2. Verify alert routing:
   ```bash
   amtool config routes show
   ```

3. Check Alertmanager logs:
   ```bash
   docker-compose logs alertmanager
   ```

### Notifications Not Sending

1. Test notification channels independently
2. Verify environment variables are set
3. Check Alertmanager notification logs
4. Verify webhook URLs and API keys
5. Test with simple alerts first

## Additional Resources

- [Prometheus Alerting Documentation](https://prometheus.io/docs/alerting/latest/overview/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
- [Alert Best Practices](https://prometheus.io/docs/practices/alerting/)
- [Runbook Examples](https://github.com/kubernetes-monitoring/kubernetes-mixin/tree/master/runbook.md)
