# ============================================================================
# GeoSense AI: Multi-Stage Production Dockerfile (CPU-Optimized)
# ============================================================================

# ----------------------------------------------------------------------------
# Stage 1: Frontend Build (Node.js)
# ----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ----------------------------------------------------------------------------
# Stage 2: Backend Runtime & Unified Server (Python 3.10)
# ----------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# Geospatial and Image Processing System Dependencies
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    DEVICE=cpu \
    TORCH_THREADS=8 \
    DEMO_MODE=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gdal-bin \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements with CPU-optimized PyTorch
COPY backend/requirements.txt ./
RUN pip install --upgrade pip && \
    pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy Backend Application & Model Registry
COPY backend/ ./backend/
COPY models/ ./models/

# Copy Built Frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create storage directories
RUN mkdir -p /app/data/uploads /app/data/processed

ENV PYTHONPATH="/app/backend"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
