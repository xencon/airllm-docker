# AirLLM Docker Setup

A complete Dockerized environment for running AirLLM, optimized for running massive Large Language Models (LLMs) on single GPUs using layer swapping, 4-bit quantization, and NVMe optimization.

This repository is specifically tailored to run on consumer hardware while providing a seamless, OpenAI-compatible streaming API.

## Features

- **OpenAI-Compatible API**: Streaming endpoint (`/v1/chat/completions`) ready for drop-in integration with various UIs and CLI tools.
- **NVMe Layer Swapping Supported**: Built-in support for blazing-fast model inference using NVMe RAM drives when massive models won't fit entirely in simple GPU VRAM.
- **Optimized for Consumer Hardware**: Built-in environment optimizations including BitsAndBytes (`nf4`), Flash Attention 2 (where supported), and OMP CPU pre-fetching logic.

## Prerequisites

- **Docker Engine**
- **NVIDIA GPU Drivers** (CUDA 12.4+ supported)
- **NVIDIA Container Toolkit** (Required to pass GPUs into the container)
- *Optional but Highly Recommended*: An NVMe Drive for faster layer swapping.

## Quick Start Configuration

### 1. Model Configuration

By default, this repository is configured to serve the `Qwen/Qwen2.5-Coder-7B-Instruct` model (as defined in `config.json`). You can customize this by editing the `config.json` file inside the repository before starting the server.

### 2. NVMe Drive Setup (Recommended)

To achieve the best inference performance via layer swapping, it is highly recommended to dedicate an NVMe drive on your host machine to `/mnt/nvme_ram`.

If you have a dedicated drive available (e.g., `/dev/nvme0n1` or `/dev/sde`), you can format and mount it using the following commands:
> **WARNING:** Formatting a drive will erase all its existing data! Be absolutely sure you have the correct drive identifier.

```bash
# Wipe any existing filesystem signatures
sudo wipefs -a <your_nvme_device>

# Format the drive
sudo mkfs.ext4 -F <your_nvme_device>

# Create mount point
sudo mkdir -p /mnt/nvme_ram

# Mount the drive
sudo mount -o noatime <your_nvme_device> /mnt/nvme_ram

# Set permissions
sudo chown -R $USER:$USER /mnt/nvme_ram
sudo chmod -R 755 /mnt/nvme_ram
```

Place your `config.json` inside `/mnt/nvme_ram`. When starting the server, if the script detects `/mnt/nvme_ram/config.json`, it will use the NVMe drive automatically. Otherwise, it will fall back to using a local `./models` directory for the model cache.

### 3. Usage

Use the provided `airllm.sh` control script to manage the container lifecycle.

**Start the Server:**
This will build the Docker image (if not already built) and run the container on port 11434.
```bash
./airllm.sh start
```

**Stop the Server:**
```bash
./airllm.sh stop
```

**Restart the Server:**
```bash
./airllm.sh restart
```

**Follow Server Logs:**
```bash
./airllm.sh logs
```

## Continue CLI Integration

This server is specifically designed to be fully compatible with the [Continue CLI](https://continue.dev/docs/reference/Model%20Providers/openai) as an OpenAI-compatible custom provider.

### Installing Continue CLI

To use the Continue CLI, you can install it via npm:

```bash
npm install -g @continuedev/cli
```

### Configuration

To connect the Continue CLI to your AirLLM server, create or modify your `~/.continue/config.json` with the following entry:

```json
{
  "models": [
    {
      "title": "AirLLM Qwen2.5-Coder",
      "provider": "openai",
      "model": "qwen2.5-coder-7b",
      "apiBase": "http://localhost:11434/v1"
    }
  ]
}
```

Now you can run the CLI tool and use your locally hosted LLM!

```bash
continue
```
