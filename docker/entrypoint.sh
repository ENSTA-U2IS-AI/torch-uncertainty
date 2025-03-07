#!/bin/bash
set -e  # Exit immediately if a command fails

echo "🚀 Starting container..."

# Ensure SSH directory exists and has correct permissions
if [ ! -d /root/.ssh ]; then
    echo "📂 Creating SSH directory..."
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
fi

# Ensure the VM's public SSH key is added for authentication
if [ -z "$VM_SSH_PUBLIC_KEY" ]; then
    echo "❌ Error: Please set the VM_SSH_PUBLIC_KEY environment variable."
    exit 1
fi
if [ ! -f /root/.ssh/authorized_keys ] || ! grep -q "$VM_SSH_PUBLIC_KEY" /root/.ssh/authorized_keys; then
    echo "🔑 Adding VM SSH public key..."
    echo "$VM_SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
fi

# Ensure GitHub SSH private key is set up for authentication
if [ -z "$GITHUB_SSH_PRIVATE_KEY" ]; then
    echo "❌ Error: Please set the GITHUB_SSH_PRIVATE_KEY environment variable."
    exit 1
fi
if [ ! -f /root/.ssh/github_rsa ]; then
    echo "🔐 Adding GitHub SSH private key..."
    echo "$GITHUB_SSH_PRIVATE_KEY" > /root/.ssh/github_rsa && chmod 600 /root/.ssh/github_rsa
fi

# Configure SSH client for GitHub authentication
if [ ! -f /root/.ssh/config ] || ! grep -q 'Host github.com' /root/.ssh/config; then
    echo "⚙️  Configuring SSH client for GitHub authentication..."
    cat <<EOF > /root/.ssh/config
Host github.com
  User git
  IdentityFile /root/.ssh/github_rsa
EOF
    chmod 600 /root/.ssh/config
fi

# Add GitHub to known hosts (to avoid SSH verification prompts)
echo "📌 Ensuring GitHub is a known host..."
ssh-keygen -F github.com > /dev/null 2>&1 || ssh-keyscan github.com >> /root/.ssh/known_hosts

# Start SSH agent and add GitHub private key (if not already added)
if ! pgrep -x "ssh-agent" > /dev/null; then
    echo "🕵️  Starting SSH agent..."
    eval "$(ssh-agent -s)"
fi
if ssh-add -l | grep -q github_rsa; then
    echo "✅ GitHub SSH key already added."
else
    echo "🔑 Adding GitHub SSH key to agent..."
    ssh-add /root/.ssh/github_rsa
fi

# Set Git user name and email (if provided)
if [ -n "$GIT_USER_NAME" ]; then
    echo "👤 Setting Git username: $GIT_USER_NAME"
    git config --global user.name "$GIT_USER_NAME"
fi
if [ -n "$GIT_USER_EMAIL" ]; then
    echo "📧 Setting Git email: $GIT_USER_EMAIL"
    git config --global user.email "$GIT_USER_EMAIL"
fi

# Ensure first-time setup runs only once
if [ ! -f /workspace/.setup_done ]; then
    echo "🛠️ Running first-time setup..."

    # Ensure GitHub username is set
    if [ -z "$GITHUB_USER" ]; then
        echo "❌ Error: Please set the GITHUB_USER environment variable."
        exit 1
    fi

    # Clone GitHub repo if not already cloned
    if [ ! -d "/workspace/.git" ]; then
        echo "📦 Cloning repository: $GITHUB_USER/torch-uncertainty..."
        git clone git@github.com:$GITHUB_USER/torch-uncertainty.git /workspace
    fi

    # Mark setup as completed
    touch /workspace/.setup_done
    echo "✅ First-time setup complete!"
else
    echo "⏩ Skipping first-time setup (already done)."
fi

# Apply compact shell prompt customization (if enabled)
if [ -n "$USE_COMPACT_SHELL_PROMPT" ]; then
    echo "🎨 Applying compact shell prompt customization..."
    echo 'force_color_prompt=yes' >> /root/.bashrc
    echo 'PS1="\[\033[01;34m\]\W\[\033[00m\]\$ "' >> /root/.bashrc
    echo 'if [ -x /usr/bin/dircolors ]; then' >> /root/.bashrc
    echo '    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"' >> /root/.bashrc
    echo '    alias ls="ls --color=auto"' >> /root/.bashrc
    echo '    alias grep="grep --color=auto"' >> /root/.bashrc
    echo '    alias fgrep="fgrep --color=auto"' >> /root/.bashrc
    echo '    alias egrep="egrep --color=auto"' >> /root/.bashrc
    echo 'fi' >> /root/.bashrc
fi

# Ensure /workspace is in PYTHONPATH
if ! echo "$PYTHONPATH" | grep -q "/workspace"; then
    echo "📌 Adding /workspace to PYTHONPATH"
    export PYTHONPATH="/workspace:$PYTHONPATH"
else
    echo "✅ PYTHONPATH is already correctly set."
fi

# Check if torch_uncertainty is installed in editable mode
if pip show torch_uncertainty | grep -q "Editable project location: /workspace"; then
    echo "✅ torch_uncertainty is already installed in editable mode. 🎉"
else
    echo "🔄 Reinstalling torch_uncertainty in editable mode..."
    pip uninstall -y torch-uncertainty
    pip install -e /workspace
    echo "✅ torch_uncertainty is now installed in editable mode! 🚀"
fi

# Activate pre-commit hooks (if enabled)
echo "🔗 Activating pre-commit hooks..."
pre-commit install

# Ensure SSH server is started
echo "🔑 Starting SSH server..."
mkdir -p /run/sshd && chmod 755 /run/sshd
/usr/sbin/sshd -D
