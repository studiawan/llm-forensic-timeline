FROM python:3.13-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install package with all extras
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[all]"

# Copy prompt and dataset templates
COPY dataset/ ./dataset/
COPY prompt/ ./prompt/

ENTRYPOINT ["llm-forensic-timeline"]
CMD ["--help"]
