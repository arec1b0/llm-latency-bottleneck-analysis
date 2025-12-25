variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "instance_type" {
  description = "EC2 instance type with GPU support"
  type        = string
  default     = "g5.xlarge" # NVIDIA A10G GPU
  
  validation {
    condition = contains([
      "g5.xlarge", "g5.2xlarge", "g5.4xlarge", "g5.8xlarge", "g5.12xlarge",
      "g5.16xlarge", "g5.24xlarge", "p3.2xlarge", "p3.8xlarge", "p3.16xlarge",
      "p4d.24xlarge", "g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge"
    ], var.instance_type)
    error_message = "Instance type must be a GPU-enabled instance."
  }
}

variable "ami_id" {
  description = "AMI ID for the instance (Ubuntu with Docker pre-installed)"
  type        = string
  default     = ""
}

variable "ssh_key_path" {
  description = "Path to SSH private key for instance access"
  type        = string
  default     = "~/.ssh/id_rsa"
}

variable "ssh_key_name" {
  description = "Name of SSH key pair in AWS"
  type        = string
  default     = "llm-inference-key"
}

variable "environment" {
  description = "Environment tag"
  type        = string
  default     = "production"
}
