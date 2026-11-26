FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ensure matplotlib & Hugging Face caches are writable under non-root Spaces user
ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=7860

# Install CPU PyTorch first for fast, lightweight Space builds
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install project & web dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, checkpoints, UI, and evaluation benchmarks
COPY server.py .
COPY src/ ./src/
COPY web/ ./web/
COPY checkpoints/ ./checkpoints/
COPY datasets/Set5/ ./datasets/Set5/
COPY datasets/Set14/ ./datasets/Set14/
COPY results/ ./results/

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
