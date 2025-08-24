# DocuChat Web UI Docker Container
# Multi-stage build for optimized production image

# Build stage - Install dependencies and prepare application
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage - Minimal runtime image
FROM python:3.11-slim as production

# Create non-root user for security
RUN groupadd -r docuchat && useradd -r -g docuchat docuchat

# Set working directory
WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/docuchat/.local

# Copy application files
COPY --chown=docuchat:docuchat . .

# Create required directories
RUN mkdir -p /app/logs /app/chroma /app/uploads && \
    chown -R docuchat:docuchat /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER docuchat

# Add local bin to PATH for user-installed packages
ENV PATH=/home/docuchat/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Expose port
EXPOSE 7860

# Default command
CMD ["python", "web/web_app.py"]