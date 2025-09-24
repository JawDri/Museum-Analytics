# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY museum_analytics/ ./museum_analytics/
COPY setup.py .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/models /workspace/plots

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV PYTHONPATH=/workspace
ENV JUPYTER_ENABLE_LAB=yes

# Create a startup script
RUN echo '#!/bin/bash\n\
cd /workspace\n\
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""' > /start.sh
RUN chmod +x /start.sh

# Default command
CMD ["/start.sh"]