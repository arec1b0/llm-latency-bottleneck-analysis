# LLM Inference GPU Deployment

This directory contains deployment configurations for running the LLM inference service on GPU instances across major cloud providers.

## Supported Cloud Providers

- **AWS** - Terraform-based deployment with GPU instances (g5, p3, p4 series)
- **GCP** - gcloud CLI deployment with NVIDIA T4 GPUs
- **Azure** - Azure CLI deployment with GPU-enabled VMs

## Quick Start

### Prerequisites

1. **Cloud Provider CLI/Tools**
   - AWS: Terraform + AWS CLI
   - GCP: Google Cloud SDK (gcloud)
   - Azure: Azure CLI

2. **SSH Key Pair**
   - Generate SSH keys: `ssh-keygen -t rsa -b 4096 -C "your-email@example.com"`

3. **Cloud Provider Account**
   - AWS: Account with appropriate IAM permissions
   - GCP: Project with Compute Engine API enabled
   - Azure: Subscription with appropriate permissions

### Deployment

```bash
# Deploy to AWS (default)
./deploy.sh --provider aws --instance-type g5.xlarge --region us-west-2

# Deploy to GCP
./deploy.sh --provider gcp --instance-type n1-standard-4 --region us-central1

# Deploy to Azure
./deploy.sh --provider azure --instance-type Standard_NC4as_T4_v3 --region eastus
```

### Deployment Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | Cloud provider (aws, gcp, azure) | aws |
| `--instance-type` | GPU instance type | g5.xlarge |
| `--region` | Cloud region | us-west-2 |
| `--ssh-key` | SSH private key path | ~/.ssh/id_rsa |
| `--environment` | Environment tag | production |

## Instance Types

### AWS GPU Instances
- `g5.xlarge` - 1x NVIDIA A10G (24GB), 4 vCPU, 16GB RAM
- `g5.2xlarge` - 1x NVIDIA A10G (24GB), 8 vCPU, 32GB RAM
- `g5.4xlarge` - 1x NVIDIA A10G (24GB), 16 vCPU, 64GB RAM
- `p3.2xlarge` - 1x NVIDIA V100 (16GB), 8 vCPU, 61GB RAM
- `p4d.24xlarge` - 8x NVIDIA A100 (40GB), 96 vCPU, 1.1TB RAM

### GCP GPU Instances
- `n1-standard-4` - 1x NVIDIA T4 (16GB), 4 vCPU, 15GB RAM
- `n1-standard-8` - 1x NVIDIA T4 (16GB), 8 vCPU, 30GB RAM
- `n1-standard-16` - 1x NVIDIA T4 (16GB), 16 vCPU, 60GB RAM

### Azure GPU Instances
- `Standard_NC4as_T4_v3` - 1x NVIDIA T4 (16GB), 4 vCPU, 28GB RAM
- `Standard_NC8as_T4_v3` - 1x NVIDIA T4 (16GB), 8 vCPU, 56GB RAM
- `Standard_NC16as_T4_v3` - 1x NVIDIA T4 (16GB), 16 vCPU, 112GB RAM

## Post-Deployment

### Access Services

After deployment, the following services will be available:

- **API**: `http://<INSTANCE_IP>:8000`
- **Grafana**: `http://<INSTANCE_IP>:3000`
- **Jaeger**: `http://<INSTANCE_IP>:16686`
- **Prometheus**: `http://<INSTANCE_IP>:9090`

### SSH Access

```bash
# AWS
ssh -i ~/.ssh/id_rsa ubuntu@<INSTANCE_IP>

# GCP
gcloud compute ssh llm-inference-gpu --zone=<REGION>-a

# Azure
ssh -i ~/.ssh/id_rsa azureuser@<INSTANCE_IP>
```

### Management Commands

On the instance, use the management script:

```bash
# Start/stop services
/opt/llm-inference/manage.sh start
/opt/llm-inference/manage.sh stop
/opt/llm-inference/manage.sh restart

# Check status
/opt/llm-inference/manage.sh status
/opt/llm-inference/manage.sh health

# View logs
/opt/llm-inference/manage.sh logs

# Create backup
/opt/llm-inference/manage.sh backup
```

## Configuration

### Environment Variables

Key environment variables in `/opt/llm-inference/.env`:

```bash
# Model configuration
MODEL_NAME=bigscience/bloom-560m
MODEL_DEVICE=cuda
MODEL_LOAD_IN_8BIT=true

# Performance tuning
MAX_CONCURRENT_GENERATIONS=4
MAX_BATCH_SIZE=32

# Monitoring
JAEGER_UI_PORT=16686
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### GPU Optimization

The deployment includes GPU optimization settings:

- NVIDIA Container Toolkit for GPU access
- CUDA memory allocation configuration
- 8-bit quantization support
- Batch processing capabilities

## Monitoring and Observability

### Grafana Dashboards

- Pre-configured LLM metrics dashboard
- Request rate, latency, and throughput visualization
- GPU memory and utilization metrics
- Error rate and alerting

### Distributed Tracing

- OpenTelemetry tracing with Jaeger
- Request flow visualization
- Performance bottleneck identification

### Metrics Collection

- Prometheus metrics collection
- Custom LLM-specific metrics (TTFT, TPOT, throughput)
- System metrics (CPU, GPU, memory)

## Cost Optimization

### Instance Selection

- **Development**: g5.xlarge or equivalent (~$1/hr)
- **Production**: g5.4xlarge or p3.2xlarge (~$3-4/hr)
- **High-throughput**: p4d.24xlarge (~$32/hr)

### Scaling Strategies

- Horizontal scaling with load balancer
- Auto-scaling based on request queue
- Spot instances for cost reduction

## Security

### Network Security

- Security groups/firewall rules
- Only necessary ports exposed
- SSH key-based authentication

### Application Security

- Environment-based configuration
- Grafana admin password randomization
- No default credentials

## Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

2. **Services not starting**
   ```bash
   /opt/llm-inference/manage.sh status
   /opt/llm-inference/manage.sh logs
   ```

3. **Memory issues**
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Adjust batch size in .env
   MAX_BATCH_SIZE=16
   ```

### Logs and Monitoring

- Application logs: `/opt/llm-inference/logs/`
- Docker logs: `docker-compose logs`
- System logs: `/var/log/syslog`

## Cleanup

### AWS

```bash
cd deploy/aws
terraform destroy -auto-approve
```

### GCP

```bash
gcloud compute instances delete llm-inference-gpu --zone=<REGION>-a
gcloud compute firewall-rules delete allow-http allow-https allow-llm-ports
```

### Azure

```bash
az group delete --name llm-inference-rg --yes
```

## Next Steps

After deployment:

1. **Test the API**: Send requests to verify functionality
2. **Configure monitoring**: Set up alerts and dashboards
3. **Load testing**: Use the provided load testing scripts
4. **Performance tuning**: Adjust batch sizes and concurrency
5. **Set up CI/CD**: Automate deployments and updates

## Support

For deployment issues:

1. Check the deployment logs
2. Verify cloud provider permissions
3. Ensure GPU quotas are sufficient
4. Review instance type availability

For application issues:

1. Check application logs
2. Verify model download and loading
3. Monitor GPU utilization
4. Review API response times
