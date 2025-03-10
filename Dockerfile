FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Install Git, OpenSSH Server, and OpenGL (PyTorch's base image already includes Conda and Pip)
RUN apt-get update && apt-get install -y \
    git \
    openssh-server \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Create an empty README.md file and an empty torch_uncertainty module to satisfy flit
RUN touch README.md && mkdir -p torch_uncertainty && touch torch_uncertainty/__init__.py

# Copy dependency file
COPY pyproject.toml /workspace/

# Install dependencies all dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Always activate Conda when opening a new terminal
RUN echo "source /opt/conda/bin/activate" >> /root/.bashrc

# Configure SSH server
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "AuthorizedKeysFile .ssh/authorized_keys" >> /etc/ssh/sshd_config

# Expose port 8888 for TensorBoard and Jupyter Notebook and port 22 for SSH
EXPOSE 8888 22

# Entrypoint script (runs every time the container starts)
COPY docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Note that if /workspace/ is a mounted volume, any files copied to /workspace/ during the build will be overwritten by the mounted volume
# This is why we copy the entrypoint script to /usr/local/bin/ instead of /workspace/
