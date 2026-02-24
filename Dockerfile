# We use the devel image because FlashAttention/AirLLM often require 
# a CUDA compiler (nvcc) to install correctly.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies needed for Python and CUDA builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev build-essential ninja-build git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Clean Python environment and install dependencies
# -----------------------------
# Note: We keep your specific versions but ensure they are built for the current CUDA
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    torch==2.4.0 \
    transformers==4.41.2 \
    accelerate==0.32.1 \
    bitsandbytes \
    fastapi==0.109.0 \
    pydantic==1.10.12 \
    uvicorn[standard]==0.23.2 \
    uvloop \
    airllm

# OPTIMIZATION: Install FlashAttention-2
# This is a 'soft' fail: if your GPU doesn't support it, the model will 
# gracefully fall back to standard attention.
RUN pip3 install flash-attn --no-build-isolation || true

# -----------------------------
# Copy application
# -----------------------------
# Create non-root user
RUN useradd -m -u 1000 appuser

# Ensure the models directory exists and appuser owns it
# AirLLM MUST be able to write sharded layers to this folder
RUN mkdir -p /app/models && chown -R appuser:appuser /app

COPY --chown=appuser:appuser server.py /app/
# config.json is optional but good to have if you are overriding defaults
COPY --chown=appuser:appuser config.json* /app/ 

# Switch to non-root user
USER appuser

# Expose server port
EXPOSE 11434

# Start FastAPI server with uvloop for higher performance
# The --loop uvloop flag tells uvicorn to use the high-speed event loop
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "11434", "--loop", "uvloop"]
