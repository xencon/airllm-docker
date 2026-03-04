#!/usr/bin/env bash
# run_airllm.sh
# Control script to build, run, and manage AirLLM container lifecycle

set -e  # Exit on error
set -u  # Treat unset variables as an error
set -o pipefail  # Catch errors in pipelines

# Get script directory early (needed for reliable paths regardless of invocation location)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ── Config ───────────────────────────────────────────────────────────────────
IMAGE_NAME="airllm-local"
CONTAINER_NAME="airllm-server"
HOST_PORT=11434
CONTAINER_PORT=11434

# Check for NVMe RAM Volume
if mountpoint -q /mnt/nvme_ram && [ -f "/mnt/nvme_ram/config.json" ]; then
    echo "⚡ NVMe RAM drive detected and populated! Using /mnt/nvme_ram for models."
    MODEL_DIR="/mnt/nvme_ram"
else
    echo "🐢 Using standard local models folder."
    MODEL_DIR="${SCRIPT_DIR}/models"  # local models folder
fi
# ─────────────────────────────────────────────────────────────────────────────

function stop_container() {
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "🛑 Stopping running container $CONTAINER_NAME..."
        docker stop "$CONTAINER_NAME" >/dev/null
    fi
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        echo "🗑 Removing container $CONTAINER_NAME..."
        docker rm "$CONTAINER_NAME" >/dev/null
    fi
}

function build_image() {
    echo "🛠 Building Docker image $IMAGE_NAME..."
    if ! docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"; then
        echo "❌ Error: Docker build failed!" >&2
        return 1
    fi
    echo "✅ Docker image built successfully"
}

function run_container() {
    echo "🚀 Starting container $CONTAINER_NAME..."
    
    local config_args=()
    if [ -f "${SCRIPT_DIR}/config.json" ]; then
        config_args+=("-v" "${SCRIPT_DIR}/config.json:/app/config.json:ro")
        echo "📄 Found config.json, mounting to container"
    fi
    
    if docker run --gpus all -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        -p "$HOST_PORT:$CONTAINER_PORT" \
        -v "$MODEL_DIR:/app/models" \
        "${config_args[@]:-}" \
        "$IMAGE_NAME" >/dev/null; then
        
        echo "✅ Container $CONTAINER_NAME is running"
        echo ""
        echo "⚠️  IMPORTANT: The server is loading the 7B parameter model from NVMe into memory."
        echo "⏳ This process takes roughly 5 to 7 minutes on consumer hardware."
        echo "🚫 The API will NOT accept connections (e.g. from Continue CLI) until loading completes."
        echo "🔬 Run './airllm.sh logs' and wait for: 'Application startup complete.'"
        echo ""
        echo "📡 Access API at http://localhost:$HOST_PORT/ once loaded."
    else
        echo "❌ Error: Failed to start container" >&2
        return 1
    fi
}

function show_logs() {
    docker logs -f "$CONTAINER_NAME"
}

# ── Script Start ─────────────────────────────────────────────────────────────
echo "🔹 AirLLM Container Control Script"

# Parse command line argument
case "${1:-}" in
    start)
        stop_container
        build_image
        run_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        run_container
        ;;
    rebuild)
        stop_container
        build_image
        run_container
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|rebuild|logs}" >&2
        exit 1
        ;;
esac
