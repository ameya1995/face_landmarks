#!/bin/bash

# Deployment script with fallback mechanism

echo "🚀 Face Landmarks App Deployment"

# Test if we can import the required modules
python3 -c "import cv2, mediapipe, streamlit; print('✅ All modules imported successfully')" 2>/dev/null

if [ $? -eq 0 ]; then
    # Try simple version first (most compatible)
    if [ -f "streamlit_app_simple.py" ]; then
        echo "✅ Starting simple optimized app"
        exec streamlit run streamlit_app_simple.py \
            --server.port=${PORT:-8501} \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --server.enableCORS=false \
            --server.enableXsrfProtection=false \
            --logger.level=error \
            --server.maxUploadSize=50
    # Fallback to full optimized version
    elif [ -f "streamlit_app_optimized.py" ]; then
        echo "✅ Starting full optimized app"
        exec streamlit run streamlit_app_optimized.py \
            --server.port=${PORT:-8501} \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --server.enableCORS=false \
            --server.enableXsrfProtection=false \
            --logger.level=error \
            --server.maxUploadSize=50
    else
        echo "✅ Starting original app"
        exec streamlit run streamlit_app.py \
            --server.port=${PORT:-8501} \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --server.enableCORS=false \
            --server.enableXsrfProtection=false \
            --logger.level=error
    fi
else
    echo "❌ Module import failed, trying original app"
    exec streamlit run streamlit_app.py \
        --server.port=${PORT:-8501} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --logger.level=error
fi
