# Multi-stage build for production deployment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN addgroup --gid 1001 --system app && \
    adduser --no-create-home --shell /bin/false --disabled-password --uid 1001 --system --group app

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml setup.py ./
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy test files
COPY tests/ tests/
COPY examples/ examples/
COPY test_*.py ./

# Change ownership
RUN chown -R app:app /app
USER app

CMD ["python", "-m", "neuro_symbolic_law.cli", "--help"]

# Production stage
FROM base as production

# Remove unnecessary packages and clean up
RUN apt-get purge -y gcc g++ libffi-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    pip cache purge

# Change ownership
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from neuro_symbolic_law import LegalProver; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "neuro_symbolic_law.cli"]

# API server stage
FROM production as api-server

# Install additional API dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy API server code
COPY api/ api/

# Expose port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]