#!/bin/bash
# run_airllm.sh
# Control script to build, run, and manage AirLLM container lifecycle

# --------- Config ---------
IMAGE_NAME="airllm-local"
CONTAINER_NAME="airllm-server"
HOST_PORT=11434
CONTAINER_PORT=11434
MODEL_DIR="${MODEL_DIR:-$(pwd)/models}"  # use environment variable if set, otherwise default to local
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
    docker run --gpus all -d \
        --name $CONTAINER_NAME \
        -p $HOST_PORT:$CONTAINER_PORT \
        -v $MODEL_DIR:/app/models \
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
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs}"
        exit 1
        ;;
esac
