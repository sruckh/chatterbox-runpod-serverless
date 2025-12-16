FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-pip \
    python3.12-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create Python 3.12 symlink for compatibility
RUN ln -s /usr/bin/python3.12 /usr/bin/python3 && \
    ln -s /usr/bin/pip3.12 /usr/bin/pip3

# Set up Python environment
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app/
WORKDIR /app

# Clone ChatterBox repository during container startup
# This ensures we have the latest code
COPY bootstrap.sh /app/bootstrap.sh
RUN chmod +x /app/bootstrap.sh

# Runpod serverless entrypoint (through bootstrap)
CMD ["/app/bootstrap.sh"]
