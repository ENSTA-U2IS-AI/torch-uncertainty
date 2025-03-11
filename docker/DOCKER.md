# :whale: Docker image for contributors

### Pre-built Docker image
1. To pull the pre-built image from Docker Hub, simply run:
    ```bash
    docker pull docker.io/tonyzamyatin/torch-uncertainty:latest
    ```

    This image includes:
    - PyTorch with CUDA support
    - OpenGL (for visualization tasks)
    - Git, OpenSSH, and all Python dependencies

    Checkout the [registry on Docker Hub](https://hub.docker.com/repository/docker/tonyzamyatin/torch-uncertainty/general) for all available images.

2. To start a container using this image, set up the necessary environment variables and run:
    ```bash
    docker run --rm -it --gpus all -p 8888:8888 -p 22:22 \
        -e VM_SSH_PUBLIC_KEY="your-public-key" \
        -e GITHUB_SSH_PRIVATE_KEY="your-github-key" \
        -e GITHUB_USER="your-github-username" \
        -e GIT_USER_EMAIL="your-git-email" \
        -e GIT_USER_NAME="your-git-name" \
        docker.io/tonyzamyatin/torch-uncertainty
    ```

    Optionally, you can also set `-e USER_COMPACT_SHELL_PROMPT="true"`  
    to make the VM's shell prompts compact and colorized.

    **Note:** Some cloud providers offer templates, in which you can preconfigure  
    in advance which Docker image to pull and which environment variables to set.  
    In this case, the provider will pull the image, set all environment variables,  
    and start the container for you.

3. Once your cloud provider has deployed the VM, it will display the host address and SSH port.  
    You can connect to the container via SSH using:
    ```bash
    ssh -i /path/to/private_key root@<VM_HOST> -p <VM_PORT>
    ```

    Replace `<VM_HOST>` and `<VM_PORT>` with the values provided by your cloud provider,  
    and `/path/to/private_key` with the private key that corresponds to `VM_SSH_PUBLIC_KEY`.

4. The container exposes port `8888` in case you want to run Jupyter Notebooks or TensorBoard.

    **Note:** The `/workspace` directory is mounted from your local machine or cloud storage,  
    so changes persist across container restarts.  
    If using a cloud provider, ensure your network volume is correctly attached to avoid losing data.

### Modifying and publishing custom Docker image

If you want to make changes to the Dockerfile, follow these steps:
1. Edit the Dockerfile to fit your needs.

2. Build the modified image:
    ```
    docker build -t my-custom-image .
    ```

3. Push to a Docker registry (if you want to use it on another VM):
    ```
    docker tag my-custom-image mydockerhubuser/my-custom-image:tag
    docker push mydockerhubuser/my-custom-image:tag
    ```
    
4. Pull the custom image onto your VM:
    ```
    docker pull mydockerhubuser/my-custom-image
    ```
    
5. Run the container using the same docker run command with the new image name.
