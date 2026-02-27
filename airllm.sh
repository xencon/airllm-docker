#!/bin/bash
# run_airllm.sh
# Control script to build, run, and manage AirLLM container lifecycle

# --------- Config ---------
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
    MODEL_DIR="$(pwd)/models"  # local models folder
fi
# --------------------------

# Function to stop and remove container
stop_container() {
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "🛑 Stopping running container $CONTAINER_NAME..."
        docker stop $CONTAINER_NAME
    fi
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        echo "🗑 Removing container $CONTAINER_NAME..."
        docker rm $CONTAINER_NAME
    fi
}

# Function to build the Docker image
build_image() {
    echo "🛠 Building Docker image $IMAGE_NAME..."
    docker build -t $IMAGE_NAME .
    if [ $? -ne 0 ]; then
        echo "❌ Docker build failed!"
        exit 1
    fi
    echo "✅ Docker image built successfully"
}

# Function to run the container
run_container() {
    echo "🚀 Starting container $CONTAINER_NAME..."
    
    CONFIG_MOUNT=""
    if [ -f "config.json" ]; then
        CONFIG_MOUNT="-v $(pwd)/config.json:/app/config.json:ro"
        echo "📄 Found config.json, mounting to container"
    fi
    
    docker run --gpus all -d \
        --name $CONTAINER_NAME \
        --restart unless-stopped \
        -p $HOST_PORT:$CONTAINER_PORT \
        -v $MODEL_DIR:/app/models \
        $CONFIG_MOUNT \
        $IMAGE_NAME
    if [ $? -eq 0 ]; then
        echo "✅ Container $CONTAINER_NAME is running"
        echo "📡 Access API at http://localhost:$HOST_PORT/"
    else
        echo "❌ Failed to start container"
        exit 1
    fi
}

# Function to show container logs
show_logs() {
    docker logs -f $CONTAINER_NAME
}

# -------------- Script Start --------------
echo "🔹 AirLLM Container Control Script"

# Parse command line argument
case "$1" in
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
        echo "Usage: $0 {start|stop|restart|rebuild|logs}"
        exit 1
        ;;
esac
