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

# Customize shell prompt (optional)
RUN if [ ! -z "$USE_COMPACT_SHELL_PROMPT" ] && [ "$USE_COMPACT_SHELL_PROMPT" = "true" ]; then \
    echo 'force_color_prompt=yes' >> /root/.bashrc && \
    # Blue working directory, no username, and no hostname, with $ at the end
    echo 'PS1="\[\033[01;34m\]\W\[\033[00m\]\$ "' >> /root/.bashrc && \
    # Colorize ls, grep, fgrep, and egrep
    echo 'if [ -x /usr/bin/dircolors ]; then' >> /root/.bashrc && \
    echo '    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"' >> /root/.bashrc && \
    echo '    alias ls="ls --color=auto"' >> /root/.bashrc && \
    echo '    alias grep="grep --color=auto"' >> /root/.bashrc && \
    echo '    alias fgrep="fgrep --color=auto"' >> /root/.bashrc && \
    echo '    alias egrep="egrep --color=auto"' >> /root/.bashrc && \
    # Automatically change to workspace directory when opening a new terminal
    echo '    cd /workspace' >> /root/.bashrc && \
    echo 'fi' >> /root/.bashrc \
    fi;

# Configure SSH server
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "AuthorizedKeysFile .ssh/authorized_keys" >> /etc/ssh/sshd_config

# Expose port 8888 for TensorBoard and Jupyter Notebook and port 22 for SSH
EXPOSE 8888 22

# Entrypoint script (runs every time the container starts)
CMD ["/bin/bash", "-c", "\
    # Create SSH directory and set permissions if not present \
        if [ ! -d /root/.ssh ]; then \
        mkdir -p /root/.ssh && chmod 700 /root/.ssh; \
    fi; \
    # Add public key for RunPod-Auth if not present \
    if [ -z \"$RUNPOD_SSH_PUBLIC_KEY\" ]; then \
        echo 'Please set the RUNPOD_SSH_PUBLIC_KEY environment variable.'; \
        exit 1; \
    fi; \
    if [ ! -f /root/.ssh/authorized_keys ] || ! grep -q \"$RUNPOD_SSH_PUBLIC_KEY\" /root/.ssh/authorized_keys; then \
        echo \"$RUNPOD_SSH_PUBLIC_KEY\" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys; \
    fi; \
    # Add private key for GitHub-Auth if not present \
    if [ -z \"$GITHUB_SSH_PRIVATE_KEY\" ]; then \
        echo 'Please set the GITHUB_SSH_PRIVATE_KEY environment variable.'; \
        exit 1; \
    fi; \
    if [ ! -f /root/.ssh/github_rsa ]; then \
        echo \"$GITHUB_SSH_PRIVATE_KEY\" > /root/.ssh/github_rsa && chmod 600 /root/.ssh/github_rsa; \
    fi; \
    # Add GitHub credentials to SSH config if not present \
    if [ ! -f /root/.ssh/config ] || ! grep -q 'Host github.com' /root/.ssh/config; then \
        echo 'Host github.com' > /root/.ssh/config; \
        echo '  User git' >> /root/.ssh/config; \
        echo '  IdentityFile /root/.ssh/github_rsa' >> /root/.ssh/config; \
        chmod 600 /root/.ssh/config; \
    fi; \
    # Add GitHub to known hosts if not already added \
    ssh-keygen -F github.com > /dev/null 2>&1 || ssh-keyscan github.com >> /root/.ssh/known_hosts; \
    # Start SSH agent if not running and add GitHub private key \
    if ! pgrep -x \"ssh-agent\" > /dev/null; then \
        eval $(ssh-agent -s); \
    fi; \
    ssh-add -l | grep github_rsa > /dev/null || ssh-add /root/.ssh/github_rsa; \
    # Ensure first-time setup only runs once \
    if [ ! -f /workspace/.setup_done ]; then \
        echo 'Running first-time setup...'; \
        # Clone GitHub repo if not already cloned \
        if [ -z \"$GITHUB_USER\" ]; then \
            echo 'Please set the GITHUB_USER environment variable.'; \
            exit 1; \
        fi; \
        if [ ! -d \"/workspace/.git\" ]; then \
            git clone git@github.com:$GITHUB_USER/torch-uncertainty.git /workspace; \
        fi; \
        # Set Git user name and email if provided \
        if [ ! -z \"$GIT_USER_NAME\" ]; then \
            git config --global user.name \"$GIT_USER_NAME\"; \
        fi; \
        if [ ! -z \"$GIT_USER_EMAIL\" ]; then \
            git config --global user.email \"$GIT_USER_EMAIL\"; \
        fi; \
        # Mark first-time setup as done \
        touch /workspace/.setup_done; \
    else \
        echo 'Skipping first-time setup, already done.'; \
    fi; \
    # Start SSH server \
    mkdir -p /run/sshd && chmod 755 /run/sshd; \
    /usr/sbin/sshd -D"]
