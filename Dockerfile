# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for MetaDrive and other Python packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-xcb1 \
    libx11-6 \
    libglu1-mesa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the MetaDrive repository
RUN git clone https://github.com/metadriverse/metadrive.git /app/metadrive

# Copy all local files into the container
COPY . /app

# Create base directory for saving trained models and outputs
RUN mkdir -p /app/trained_model

# Install additional dependencies from requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Add the MetaDrive directory to the PYTHONPATH
ENV PYTHONPATH="/app:/app/metadrive:${PYTHONPATH}"

# Expose any required ports (if needed for the application)
EXPOSE 8080

# Run the training script on container startup
CMD ["python", "train.py"]
