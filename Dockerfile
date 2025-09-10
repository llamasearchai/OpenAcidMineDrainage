# OpenAcidMineDrainage Dockerfile - Optimized for Orbstack
FROM python:3.11-slim as base

# Set environment variables for better Python performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash oamd

# Set work directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Copy source code
COPY src/ ./src/

# Create artifacts directory
RUN mkdir -p artifacts && chown -R oamd:oamd /app

# Switch to non-root user
USER oamd

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import oamd; print('OpenAcidMineDrainage is healthy')" || exit 1

# Default command
CMD ["oamd", "--help"]

# Multi-stage build for production
FROM base as production

# Copy all necessary files
COPY data/ ./data/
COPY tests/ ./tests/
COPY README.md LICENSE ./

# Install additional production dependencies if needed
RUN pip install --only-binary=all -e ".[test]"

# Set production environment
ENV PYTHONPATH=/app/src \
    LOG_LEVEL=INFO

# Expose any ports if needed (none currently)
# EXPOSE 8000

# Production command
CMD ["oamd", "--version"]
