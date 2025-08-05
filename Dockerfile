FROM python:3.10-slim


WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Ensure model and preprocess directories exist and are properly copied
RUN mkdir -p backend/model backend/preprocess

EXPOSE 8080

# Production WSGI server — 1 worker is fine for free plan
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "backend.app:app"]
