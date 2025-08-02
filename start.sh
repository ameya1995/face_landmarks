#!/bin/bash

# Set environment variables to suppress common warnings and optimize memory
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS=ignore::UserWarning
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

# Start Streamlit with memory-optimized settings
exec streamlit run streamlit_app_optimized.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --logger.level=error \
    --server.maxUploadSize=50 \
    --server.maxMessageSize=50
