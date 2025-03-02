FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

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

# Expose port 8888 for TensorBoard and Jupyter Notebook
EXPOSE 8888

CMD [ "/bin/bash" ]
