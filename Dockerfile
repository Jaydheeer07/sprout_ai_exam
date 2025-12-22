FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for all services
EXPOSE 8000 8001 8080

# Default command (overridden in docker-compose for each service)
CMD ["uvicorn", "task2_sentiment_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
