# --- STAGE 1: BUILDER ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

WORKDIR /build

# Install system dependencies needed for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev build-essential ninja-build git \
    && rm -rf /var/lib/apt/lists/*

# Install build-time dependencies
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

# OPTIMIZATION: Build Flash-Attention 2
# We set MAX_JOBS to prevent the laptop from freezing during the heavy compile
RUN MAX_JOBS=4 pip3 install flash-attn --no-build-isolation || true

# --- STAGE 2: RUNTIME ---
# We switch to 'runtime' which is much smaller and safer
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Install minimal runtime python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire python environment from the builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Setup User and Permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Copy Application Files
COPY --chown=appuser:appuser server.py /app/
COPY --chown=appuser:appuser config.json* /app/

# Environment Optimizations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # AirLLM works best when it can utilize multiple CPU threads for layer pre-fetching
    OMP_NUM_THREADS=8 

USER appuser
EXPOSE 11434

# Use the standard loop for uvicorn which triggers uvloop automatically 
# when installed via uvicorn[standard]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "11434", "--loop", "uvloop", "--http", "httptools"]
