# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p chroma docs/s1 docs/s2 docs/s3

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose ports
EXPOSE 8501 11434

# Create startup script
RUN echo '#!/bin/bash\n\
# Start Ollama service in background\n\
ollama serve &\n\
\n\
# Wait for Ollama to be ready\n\
echo "Waiting for Ollama to start..."\n\
sleep 10\n\
\n\
# Pull the model if not exists\n\
ollama pull llama3.2 || echo "Model already exists or will be pulled on first use"\n\
\n\
# Start the application based on command\n\
if [ "$1" = "streamlit" ]; then\n\
    echo "Starting Streamlit app..."\n\
    streamlit run llm_processing.py\n\
elif [ "$1" = "load-docs" ]; then\n\
    echo "Loading documents..."\n\
    python load_docs.py\n\
elif [ "$1" = "cli" ]; then\n\
    echo "Starting CLI mode..."\n\
    python llm_processing.py\n\
else\n\
    echo "Usage: docker run <image> [streamlit|load-docs|cli]"\n\
    echo "Default: streamlit"\n\
    streamlit run llm_processing.py\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["/app/start.sh", "streamlit"] 