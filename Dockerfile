# Base image with PyTorch + CUDA 12.1
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SAFETENSORS_FAST_LOAD=1

# Upgrade pip
RUN pip install --upgrade pip

# Install git (needed for git+https installs if any)
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# -----------------------------
# Clean Python environment and install dependencies
# -----------------------------
RUN pip uninstall -y torch torchvision transformers || true

RUN pip install --no-cache-dir \
    torch==2.4.0 \
    transformers==4.41.2 \
    accelerate==0.32.1 \
    bitsandbytes \
    fastapi==0.109.0 \
    uvicorn==0.23.2 \
    pydantic==1.10.12

# -----------------------------
# Copy application
# -----------------------------
# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

COPY --chown=appuser:appuser server.py /app/
COPY --chown=appuser:appuser config.json /app/
COPY --chown=appuser:appuser models /app/models

# Switch to non-root user
USER appuser

# Expose server port
EXPOSE 11434

# Start FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "11434"]
