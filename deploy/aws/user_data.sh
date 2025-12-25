#!/bin/bash
set -e

# Log all output
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting LLM Inference instance setup..."

# Update system packages
apt-get update
apt-get upgrade -y

# Install essential packages
apt-get install -y \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    python3-venv \
    htop \
    jq \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker
echo "Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Install Docker Compose
echo "Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Install NVIDIA Container Toolkit for GPU support
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
systemctl restart docker

# Create application directory
mkdir -p /opt/llm-inference
cd /opt/llm-inference

# Clone the repository (or copy files if already available)
echo "Setting up application files..."
if [ ! -d "/opt/llm-inference/src" ]; then
    # This assumes the code is copied during deployment
    # In practice, you might use AWS CodeDeploy, S3, or git clone
    echo "Please ensure application code is copied to /opt/llm-inference"
fi

# Create docker-compose.yml from template
echo "${docker_compose_content}" | base64 -d > /opt/llm-inference/docker-compose.yml

# Create .env file from template
echo "${env_content}" | base64 -d > /opt/llm-inference/.env

# Update environment variables for production
cat >> /opt/llm-inference/.env << EOF

# Production environment variables
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000
MAX_CONCURRENT_GENERATIONS=4

# GPU settings
MODEL_DEVICE=cuda
MODEL_LOAD_IN_8BIT=true

# Monitoring endpoints (accessible from internet)
JAEGER_UI_PORT=16686
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Security
GF_SECURITY_ADMIN_PASSWORD=$(openssl rand -base64 32)
GF_USERS_ALLOW_SIGN_UP=false

# Performance
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EOF

# Set proper permissions
chown -R ubuntu:ubuntu /opt/llm-inference
chmod +x /opt/llm-inference/.env

# Create systemd service for Docker Compose
echo "Creating systemd service..."
cat > /etc/systemd/system/llm-inference.service << EOF
[Unit]
Description=LLM Inference Stack
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/llm-inference
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
systemctl daemon-reload
systemctl enable llm-inference.service
systemctl start llm-inference.service

# Wait for services to start
echo "Waiting for services to start..."
sleep 30

# Check if services are running
echo "Checking service status..."
docker-compose ps

# Install monitoring and health check scripts
cat > /opt/llm-inference/health-check.sh << 'EOF'
#!/bin/bash
# Health check script for LLM Inference API

API_URL="http://localhost:8000"
HEALTH_ENDPOINT="$API_URL/health"

# Check API health
response=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_ENDPOINT")

if [ "$response" = "200" ]; then
    echo "API is healthy"
    exit 0
else
    echo "API is unhealthy (HTTP $response)"
    exit 1
fi
EOF

chmod +x /opt/llm-inference/health-check.sh

# Create log rotation for Docker containers
cat > /etc/logrotate.d/llm-inference << EOF
/opt/llm-inference/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
    postrotate
        docker-compose restart > /dev/null 2>&1 || true
    endscript
}
EOF

# Create monitoring dashboard setup script
cat > /opt/llm-inference/setup-monitoring.sh << 'EOF'
#!/bin/bash
# Setup monitoring dashboards and alerts

echo "Setting up monitoring..."

# Wait for Grafana to be ready
until curl -s http://localhost:3000/api/health > /dev/null; do
    echo "Waiting for Grafana..."
    sleep 5
done

# Import dashboard (if API is available)
# This would require Grafana API setup
echo "Grafana is ready at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):3000"
echo "Default admin password is in /opt/llm-inference/.env"

echo "Monitoring setup complete!"
EOF

chmod +x /opt/llm-inference/setup-monitoring.sh

# Create backup script
cat > /opt/llm-inference/backup.sh << 'EOF'
#!/bin/bash
# Backup script for configurations and data

BACKUP_DIR="/opt/llm-inference/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/llm-inference-backup-$DATE.tar.gz"

mkdir -p "$BACKUP_DIR"

# Backup configurations and data
tar -czf "$BACKUP_FILE" \
    .env \
    docker-compose.yml \
    grafana/ \
    prometheus/ \
    logs/ \
    --exclude='logs/*.log'

echo "Backup created: $BACKUP_FILE"

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "llm-inference-backup-*.tar.gz" -mtime +7 -delete
EOF

chmod +x /opt/llm-inference/backup.sh

# Add cron job for daily backup
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/llm-inference/backup.sh") | crontab -

# Create startup script for easy management
cat > /opt/llm-inference/manage.sh << 'EOF'
#!/bin/bash
# Management script for LLM Inference stack

case "$1" in
    start)
        echo "Starting LLM Inference stack..."
        systemctl start llm-inference.service
        ;;
    stop)
        echo "Stopping LLM Inference stack..."
        systemctl stop llm-inference.service
        ;;
    restart)
        echo "Restarting LLM Inference stack..."
        systemctl restart llm-inference.service
        ;;
    status)
        echo "LLM Inference stack status:"
        systemctl status llm-inference.service
        docker-compose ps
        ;;
    logs)
        echo "Showing logs..."
        docker-compose logs -f
        ;;
    health)
        echo "Checking health..."
        /opt/llm-inference/health-check.sh
        ;;
    backup)
        echo "Creating backup..."
        /opt/llm-inference/backup.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|health|backup}"
        exit 1
        ;;
esac
EOF

chmod +x /opt/llm-inference/manage.sh

# Print deployment information
echo "=========================================="
echo "LLM Inference Deployment Complete!"
echo "=========================================="
echo "Instance IP: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo ""
echo "Services:"
echo "  API: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "  Grafana: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):3000"
echo "  Jaeger: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):16686"
echo "  Prometheus: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):9090"
echo ""
echo "Management commands:"
echo "  /opt/llm-inference/manage.sh {start|stop|restart|status|logs|health|backup}"
echo ""
echo "Grafana admin password is in /opt/llm-inference/.env"
echo "=========================================="

# Create a marker file to indicate successful setup
touch /opt/llm-inference/.setup-complete

echo "Setup completed successfully!"
