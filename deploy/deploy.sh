#!/bin/bash
set -e

# Deployment script for LLM Inference on GPU instances
# Supports AWS, GCP, and Azure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CLOUD_PROVIDER="aws"
INSTANCE_TYPE="g5.xlarge"
REGION="us-west-2"
SSH_KEY_PATH="$HOME/.ssh/id_rsa"
ENVIRONMENT="production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)
            CLOUD_PROVIDER="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --ssh-key)
            SSH_KEY_PATH="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --provider PROVIDER     Cloud provider (aws, gcp, azure) [default: aws]"
            echo "  --instance-type TYPE    GPU instance type [default: g5.xlarge]"
            echo "  --region REGION          Cloud region [default: us-west-2]"
            echo "  --ssh-key PATH          SSH private key path [default: ~/.ssh/id_rsa]"
            echo "  --environment ENV       Environment [default: production]"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --provider aws --instance-type g5.xlarge"
            echo "  $0 --provider gcp --instance-type n1-standard-4 --region us-central1"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
validate_inputs() {
    log_info "Validating deployment parameters..."
    
    if [[ ! "$CLOUD_PROVIDER" =~ ^(aws|gcp|azure)$ ]]; then
        log_error "Invalid cloud provider: $CLOUD_PROVIDER"
        exit 1
    fi
    
    if [[ ! -f "$SSH_KEY_PATH" ]]; then
        log_error "SSH key not found: $SSH_KEY_PATH"
        exit 1
    fi
    
    log_info "Validation complete"
}

# Deploy to AWS
deploy_aws() {
    log_info "Deploying to AWS..."
    
    cd "$SCRIPT_DIR/aws"
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install Terraform first."
        exit 1
    fi
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init
    
    # Plan deployment
    log_info "Planning deployment..."
    terraform plan \
        -var="instance_type=$INSTANCE_TYPE" \
        -var="aws_region=$REGION" \
        -var="ssh_key_path=$SSH_KEY_PATH" \
        -var="environment=$ENVIRONMENT"
    
    # Apply deployment
    log_info "Applying deployment..."
    terraform apply \
        -var="instance_type=$INSTANCE_TYPE" \
        -var="aws_region=$REGION" \
        -var="ssh_key_path=$SSH_KEY_PATH" \
        -var="environment=$ENVIRONMENT" \
        -auto-approve
    
    # Get outputs
    INSTANCE_IP=$(terraform output -raw instance_public_ip)
    INSTANCE_ID=$(terraform output -raw instance_id)
    
    log_info "AWS deployment complete!"
    log_info "Instance IP: $INSTANCE_IP"
    log_info "Instance ID: $INSTANCE_ID"
    
    # Wait for instance to be ready
    log_info "Waiting for instance to be ready..."
    sleep 60
    
    # Copy application files
    log_info "Copying application files..."
    scp -i "$SSH_KEY_PATH" -r "$PROJECT_ROOT/src" ubuntu@"$INSTANCE_IP":/tmp/
    scp -i "$SSH_KEY_PATH" -r "$PROJECT_ROOT/docker" ubuntu@"$INSTANCE_IP":/tmp/
    scp -i "$SSH_KEY_PATH" "$PROJECT_ROOT/requirements.txt" ubuntu@"$INSTANCE_IP":/tmp/
    scp -i "$SSH_KEY_PATH" "$PROJECT_ROOT/.env.example" ubuntu@"$INSTANCE_IP":/tmp/
    
    # Move files to correct location
    ssh -i "$SSH_KEY_PATH" ubuntu@"$INSTANCE_IP" "sudo mkdir -p /opt/llm-inference && sudo mv /tmp/* /opt/llm-inference/ && sudo chown -R ubuntu:ubuntu /opt/llm-inference"
    
    # Install Python dependencies and start services
    ssh -i "$SSH_KEY_PATH" ubuntu@"$INSTANCE_IP" "cd /opt/llm-inference && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && sudo systemctl restart llm-inference.service"
    
    echo "$INSTANCE_IP" > "$SCRIPT_DIR/.instance_ip"
    echo "$INSTANCE_ID" > "$SCRIPT_DIR/.instance_id"
}

# Deploy to GCP
deploy_gcp() {
    log_info "Deploying to GCP..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud SDK is not installed. Please install gcloud first."
        exit 1
    fi
    
    # Set project and region
    gcloud config set project "$(gcloud config get-value project)"
    gcloud config set compute/region "$REGION"
    
    # Create instance
    log_info "Creating GCP instance..."
    gcloud compute instances create llm-inference-gpu \
        --zone="${REGION}-a" \
        --machine-type="$INSTANCE_TYPE" \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --image-family="ubuntu-2004-lts" \
        --image-project="ubuntu-os-cloud" \
        --boot-disk-size=100GB \
        --boot-disk-type="pd-ssd" \
        --tags="http-server,https-server,llm-inference" \
        --metadata-from-file="user-data=$SCRIPT_DIR/gcp/user_data.sh" \
        --scopes="cloud-platform"
    
    # Create firewall rules
    log_info "Creating firewall rules..."
    gcloud compute firewall-rules create allow-http \
        --allow=tcp:80 \
        --source-ranges="0.0.0.0/0" \
        --target-tags="http-server"
    
    gcloud compute firewall-rules create allow-https \
        --allow=tcp:443 \
        --source-ranges="0.0.0.0/0" \
        --target-tags="https-server"
    
    gcloud compute firewall-rules create allow-llm-ports \
        --allow=tcp:8000,tcp:3000,tcp:16686,tcp:9090 \
        --source-ranges="0.0.0.0/0" \
        --target-tags="llm-inference"
    
    # Get instance IP
    INSTANCE_IP=$(gcloud compute instances describe llm-inference-gpu \
        --zone="${REGION}-a" \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    
    log_info "GCP deployment complete!"
    log_info "Instance IP: $INSTANCE_IP"
    
    echo "$INSTANCE_IP" > "$SCRIPT_DIR/.instance_ip"
}

# Deploy to Azure
deploy_azure() {
    log_info "Deploying to Azure..."
    
    # Check if az is installed
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed. Please install Azure CLI first."
        exit 1
    fi
    
    # Create resource group
    log_info "Creating resource group..."
    az group create \
        --name "llm-inference-rg" \
        --location "$REGION"
    
    # Create VM
    log_info "Creating Azure VM..."
    az vm create \
        --resource-group "llm-inference-rg" \
        --name "llm-inference-vm" \
        --image "UbuntuLTS" \
        --size "$INSTANCE_TYPE" \
        --admin-username "azureuser" \
        --ssh-key-values "$SSH_KEY_PATH.pub" \
        --custom-data "$SCRIPT_DIR/azure/user_data.sh" \
        --nsg "" \
        --public-ip-sku Standard
    
    # Open ports
    log_info "Opening ports..."
    az vm open-port \
        --resource-group "llm-inference-rg" \
        --name "llm-inference-vm" \
        --port 8000
    
    az vm open-port \
        --resource-group "llm-inference-rg" \
        --name "llm-inference-vm" \
        --port 3000
    
    az vm open-port \
        --resource-group "llm-inference-rg" \
        --name "llm-inference-vm" \
        --port 16686
    
    # Get instance IP
    INSTANCE_IP=$(az vm show \
        --resource-group "llm-inference-rg" \
        --name "llm-inference-vm" \
        --show-details \
        --query "publicIps" \
        --output tsv)
    
    log_info "Azure deployment complete!"
    log_info "Instance IP: $INSTANCE_IP"
    
    echo "$INSTANCE_IP" > "$SCRIPT_DIR/.instance_ip"
}

# Show deployment info
show_deployment_info() {
    if [[ -f "$SCRIPT_DIR/.instance_ip" ]]; then
        INSTANCE_IP=$(cat "$SCRIPT_DIR/.instance_ip")
        log_info "Deployment Information:"
        log_info "  Instance IP: $INSTANCE_IP"
        log_info "  API URL: http://$INSTANCE_IP:8000"
        log_info "  Grafana: http://$INSTANCE_IP:3000"
        log_info "  Jaeger: http://$INSTANCE_IP:16686"
        log_info "  Prometheus: http://$INSTANCE_IP:9090"
        log_info ""
        log_info "To connect to the instance:"
        if [[ "$CLOUD_PROVIDER" == "aws" ]]; then
            log_info "  ssh -i $SSH_KEY_PATH ubuntu@$INSTANCE_IP"
        elif [[ "$CLOUD_PROVIDER" == "gcp" ]]; then
            log_info "  gcloud compute ssh llm-inference-gpu --zone=${REGION}-a"
        elif [[ "$CLOUD_PROVIDER" == "azure" ]]; then
            log_info "  ssh -i $SSH_KEY_PATH azureuser@$INSTANCE_IP"
        fi
    fi
}

# Main deployment flow
main() {
    log_info "Starting LLM Inference deployment..."
    log_info "Provider: $CLOUD_PROVIDER"
    log_info "Instance Type: $INSTANCE_TYPE"
    log_info "Region: $REGION"
    log_info "Environment: $ENVIRONMENT"
    
    validate_inputs
    
    case "$CLOUD_PROVIDER" in
        aws)
            deploy_aws
            ;;
        gcp)
            deploy_gcp
            ;;
        azure)
            deploy_azure
            ;;
        *)
            log_error "Unsupported cloud provider: $CLOUD_PROVIDER"
            exit 1
            ;;
    esac
    
    show_deployment_info
    
    log_info "Deployment complete!"
}

# Run main function
main "$@"
