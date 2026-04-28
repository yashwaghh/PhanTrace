FROM python:3.11-slim

# Install system dependencies required by moviepy, librosa, scikit-image, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects PORT; default to 8080 if not set
ENV PORT=8080

# Create uploads directory (ephemeral – replace with GCS for production)
RUN mkdir -p uploads

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
