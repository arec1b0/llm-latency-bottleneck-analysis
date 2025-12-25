terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and networking
resource "aws_vpc" "llm_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "llm-inference-vpc"
  }
}

resource "aws_subnet" "llm_subnet" {
  vpc_id                  = aws_vpc.llm_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name = "llm-inference-subnet"
  }
}

resource "aws_internet_gateway" "llm_igw" {
  vpc_id = aws_vpc.llm_vpc.id

  tags = {
    Name = "llm-inference-igw"
  }
}

resource "aws_route_table" "llm_rt" {
  vpc_id = aws_vpc.llm_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.llm_igw.id
  }

  tags = {
    Name = "llm-inference-rt"
  }
}

resource "aws_route_table_association" "llm_rta" {
  subnet_id      = aws_subnet.llm_subnet.id
  route_table_id = aws_route_table.llm_rt.id
}

# Security group
resource "aws_security_group" "llm_sg" {
  name        = "llm-inference-sg"
  description = "Allow HTTP, HTTPS, SSH, and monitoring traffic"
  vpc_id      = aws_vpc.llm_vpc.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Grafana"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Jaeger UI"
    from_port   = 16686
    to_port     = 16686
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "llm-inference-sg"
  }
}

# IAM role for EC2 instance
resource "aws_iam_role" "llm_ec2_role" {
  name = "llm-inference-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "llm_ssm_policy" {
  role       = aws_iam_role.llm_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "llm_ec2_profile" {
  name = "llm-inference-ec2-profile"
  role = aws_iam_role.llm_ec2_role.name
}

# GPU instance
resource "aws_instance" "llm_gpu_instance" {
  ami           = var.ami_id
  instance_type = var.instance_type
  subnet_id     = aws_subnet.llm_subnet.id
  vpc_security_group_ids = [aws_security_group.llm_sg.id]
  iam_instance_profile = aws_iam_instance_profile.llm_ec2_profile.name

  root_block_device {
    volume_type = "gp3"
    volume_size = 100
    encrypted   = true
  }

  tags = {
    Name = "llm-inference-gpu"
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    docker_compose_content = filebase64("${path.module}/../../docker/docker-compose.yml")
    env_content = filebase64("${path.module}/../../.env.example")
  }))
}

# Elastic IP for static IP
resource "aws_eip" "llm_eip" {
  instance = aws_instance.llm_gpu_instance.id
  vpc      = true

  tags = {
    Name = "llm-inference-eip"
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

# Outputs
output "instance_public_ip" {
  description = "Public IP address of the GPU instance"
  value       = aws_eip.llm_eip.public_ip
}

output "instance_id" {
  description = "ID of the GPU instance"
  value       = aws_instance.llm_gpu_instance.id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${var.ssh_key_path} ubuntu@${aws_eip.llm_eip.public_ip}"
}
