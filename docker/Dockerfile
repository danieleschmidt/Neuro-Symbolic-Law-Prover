# Production Dockerfile for Neuro-Symbolic Legal Reasoning System
# Multi-stage build for optimized production deployment

# Stage 1: Build environment
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create build user
RUN useradd -m -u 1001 builder
USER builder
WORKDIR /home/builder

# Copy requirements and install Python dependencies
COPY --chown=builder:builder requirements.txt ./
COPY --chown=builder:builder pyproject.toml ./
COPY --chown=builder:builder setup.py ./

# Create virtual environment and install dependencies
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=builder:builder src/ ./src/
COPY --chown=builder:builder README.md ./
COPY --chown=builder:builder LICENSE ./

# Install the package
RUN . venv/bin/activate && pip install -e .

# Stage 2: Production runtime
FROM python:3.11-slim as runtime

# Set metadata labels
LABEL maintainer="Terragon Labs <support@terragonlabs.com>"
LABEL version="${BUILD_VERSION}"
LABEL description="Neuro-Symbolic Legal Reasoning System"
LABEL build-date="${BUILD_DATE}"
LABEL vcs-ref="${VCS_REF}"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Runtime libraries
    libgcc1 \
    libstdc++6 \
    # System monitoring
    procps \
    htop \
    # Network utilities
    curl \
    wget \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Create application user
RUN groupadd -r nslapp && \
    useradd -r -g nslapp -u 1001 nslapp && \
    mkdir -p /app /app/logs /app/data /app/cache && \
    chown -R nslapp:nslapp /app

# Switch to application user
USER nslapp
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=nslapp:nslapp /home/builder/venv ./venv

# Copy application code
COPY --from=builder --chown=nslapp:nslapp /home/builder/src ./src
COPY --from=builder --chown=nslapp:nslapp /home/builder/README.md ./
COPY --from=builder --chown=nslapp:nslapp /home/builder/LICENSE ./

# Copy configuration files
COPY --chown=nslapp:nslapp config/production.yaml ./config/
COPY --chown=nslapp:nslapp docker/entrypoint.sh ./
COPY --chown=nslapp:nslapp docker/healthcheck.py ./

# Make scripts executable
RUN chmod +x entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/venv/bin:$PATH"

# Default configuration
ENV NSL_WORKERS=8
ENV NSL_MAX_WORKERS=32
ENV NSL_ENABLE_AUTO_SCALING=true
ENV NSL_CACHE_SIZE=10000
ENV NSL_MAX_CACHE_SIZE=100000
ENV NSL_ENABLE_ADAPTIVE_CACHE=true
ENV NSL_ENABLE_MONITORING=true
ENV NSL_LOG_LEVEL=INFO
ENV NSL_BIND_HOST=0.0.0.0
ENV NSL_BIND_PORT=8000
ENV NSL_METRICS_PORT=9090

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python healthcheck.py || exit 1

# Volume for persistent data
VOLUME ["/app/data", "/app/logs", "/app/cache"]

# Entry point
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["serve"]