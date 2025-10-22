# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_REPO=bosonai/higgs-audio-v2-generation-3B-base
ENV AUDIO_TOKENIZER_REPO=bosonai/higgs-audio-v2-tokenizer

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install higgs-audio dependencies
RUN pip install -r https://raw.githubusercontent.com/reinSPQR/higgs-audio/main/requirements.txt

# Install repo own dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application files
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set the default command
CMD ["python", "apis.py"]