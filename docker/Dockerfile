FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# make "python" point to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# install torch + torchvision matching your original versions
RUN pip install \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# now your original list works, incl. numpy==1.23.1
RUN pip install \
    numpy==1.23.1 \
    gradio==3.16.2 \
    albumentations==1.3.0 \
    opencv-contrib-python==4.3.0.36 \
    imageio==2.9.0 \
    imageio-ffmpeg==0.4.2 \
    pytorch-lightning==1.5.0 \
    omegaconf==2.1.1 \
    "test-tube>=0.7.5" \
    streamlit==1.12.1 \
    einops==0.3.0 \
    transformers==4.19.2 \
    webdataset==0.2.5 \
    kornia==0.6 \
    open_clip_torch==2.0.2 \
    "invisible-watermark>=0.1.5" \
    streamlit-drawable-canvas==0.8.0 \
    torchmetrics==0.6.0 \
    timm==0.6.12 \
    addict==2.4.0 \
    yapf==0.32.0 \
    prettytable==3.6.0 \
    safetensors==0.2.7 \
    basicsr==1.4.2
