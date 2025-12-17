FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/chatterbox/hf_home \
    HF_HUB_CACHE=/runpod-volume/chatterbox/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git ca-certificates curl build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /workspace/chatterbox

# Copy minimal bootstrap assets; ChatterBox code is cloned in bootstrap.sh
COPY requirements.txt /workspace/chatterbox/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY bootstrap.sh /workspace/chatterbox/bootstrap.sh
COPY handler.py /workspace/chatterbox/handler.py
COPY inference.py /workspace/chatterbox/inference.py
COPY config.py /workspace/chatterbox/config.py

CMD ["bash", "/workspace/chatterbox/bootstrap.sh"]
