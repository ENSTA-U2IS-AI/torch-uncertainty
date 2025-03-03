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

# Copy README.md and torch_uncertainty module (required by pyproject.toml, otherwise flit build will fail)
COPY README.md /workspace/
COPY torch_uncertainty /workspace/torch_uncertainty

# Copy dependency file
COPY pyproject.toml /workspace/

# Install dependencies
RUN pip install --no-cache-dir ".[all]"

# Always activate Conda when opening a new terminal
RUN echo "source /opt/conda/bin/activate" >> /root/.bashrc

# Customize shell prompt
RUN echo 'force_color_prompt=yes' >> /root/.bashrc && \
    echo 'PS1="\[\033[01;34m\]\W\[\033[00m\]\$ "' >> /root/.bashrc && \
    echo 'if [ -x /usr/bin/dircolors ]; then' >> /root/.bashrc && \
    echo '    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"' >> /root/.bashrc && \
    echo '    alias ls="ls --color=auto"' >> /root/.bashrc && \
    echo '    alias grep="grep --color=auto"' >> /root/.bashrc && \
    echo '    alias fgrep="fgrep --color=auto"' >> /root/.bashrc && \
    echo '    alias egrep="egrep --color=auto"' >> /root/.bashrc && \
    echo '    cd /workspace' >> /root/.bashrc && \
    echo 'fi' >> /root/.bashrc

# Configure SSH server
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "AuthorizedKeysFile .ssh/authorized_keys" >> /etc/ssh/sshd_config

# Expose port 8888 for TensorBoard and Jupyter Notebook and port 22 for SSH
EXPOSE 8888 22

# Ensure public key for RunPod-Auth is added when the container starts
CMD ["/bin/bash", "-c", "\
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    echo \"$RUNPOD_SSH_PUBLIC_KEY\" > /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    echo \"$GITHUB_SSH_PRIVATE_KEY\" > /root/.ssh/github_rsa && \
    chmod 600 /root/.ssh/github_rsa && \
    echo 'Host github.com' > /root/.ssh/config && \
    echo '  User git' >> /root/.ssh/config && \
    echo '  IdentityFile /root/.ssh/github_rsa' >> /root/.ssh/config && \
    chmod 600 /root/.ssh/config && \
    eval $(ssh-agent -s) && ssh-add /root/.ssh/github_rsa && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    git clone git@github.com:$GITHUB_USER/torch-uncertainty.git /workspace && \
    mkdir -p /run/sshd && \
    /usr/sbin/sshd -D"]
