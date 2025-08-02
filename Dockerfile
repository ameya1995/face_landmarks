FROM python:3.11-slim

FROM python:3.11-slim

# Install essential system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    libice6 \
    libxcb1 \
    libxau6 \
    libxdmcp6 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables to suppress warnings and optimize memory
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONWARNINGS=ignore::UserWarning
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50
ENV STREAMLIT_SERVER_MAX_MESSAGE_SIZE=50

# Make startup scripts executable
RUN chmod +x start.sh deploy.sh

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port that Streamlit runs on
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Run the optimized deployment script
CMD ["./deploy.sh"]