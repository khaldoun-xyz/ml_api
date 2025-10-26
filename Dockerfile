# Use official Python runtime as base image
FROM python:3.14-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies and curl for pixi installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash && \
    mv /root/.pixi/bin/pixi /usr/local/bin/pixi

# Copy pixi configuration files
COPY pixi.toml pixi.lock ./

# Install dependencies using pixi with locked versions
RUN pixi install --locked

# Copy the application code
COPY src/ ./src/

# Copy the trained model
COPY models/ ./models/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/').read()" || exit 1

# Run the FastAPI application using pixi's task runner
CMD ["pixi", "run", "fastapi"]
