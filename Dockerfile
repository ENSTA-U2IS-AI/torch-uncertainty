FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Install git and pip
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy README.md and torch_uncertainy module (required by pyproject.toml, othwise flit build will fail)
COPY README.md .
COPY torch_uncertainty ./torch_uncertainty

# Copy dependency file
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir .

# Install OpenSSH Server
RUN apt-get update && apt-get install -y openssh-server && rm -rf /var/lib/apt/lists/*

# Create SSH directory & keys
RUN mkdir -p /var/run/sshd && echo 'root:root' | chpasswd

# Allow root login via SSH
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Expose port 8888 for TensorBoard and Jupyter Notebook
EXPOSE 8888
# Expose port 22 for SSH
EXPOSE 22

# Ensure the SSH server starts on container launch
CMD ["/usr/sbin/sshd", "-D"]
