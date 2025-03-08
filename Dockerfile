# Use CUDA-enabled base image with Python
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent Python from writing bytecode and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    -r requirements.txt \
    git+https://github.com/openai/whisper.git@main

# Verify Whisper installation
RUN python3 -c "import whisper; print('Whisper version:', whisper.__version__)"

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Default command to run the container
CMD ["python3", "whisper_diarizer_container.py"]
