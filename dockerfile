# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Railway akan set PORT secara otomatis, jadi tidak perlu di-set manual
# ENV PORT=5001

# Install system dependencies untuk catboost dan xgboost
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data

# Expose port (Railway biasanya menggunakan 8080)
EXPOSE 8080

# Command to run the application
# Railway lebih suka menggunakan gunicorn untuk production
CMD ["python", "app.py"]