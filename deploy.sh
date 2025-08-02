#!/bin/bash

# Deployment script to choose between original and optimized versions

echo "🚀 Face Landmarks App Deployment"
echo "Choose version to deploy:"
echo "1. Original (full features)"
echo "2. Optimized (memory efficient)"

# For Render.com, we'll use the optimized version by default
# You can modify this script to choose different versions

# Use optimized version for production deployment
cp requirements_optimized.txt requirements.txt
echo "✅ Using optimized requirements"

# Check if optimized app exists, otherwise use original
if [ -f "streamlit_app_optimized.py" ]; then
    echo "✅ Using optimized Streamlit app"
    exec ./start.sh
else
    echo "⚠️  Optimized app not found, using original"
    exec streamlit run streamlit_app.py \
        --server.port=${PORT:-8501} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --logger.level=error
fi
