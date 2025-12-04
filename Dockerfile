# Base Python image
FROM python:3.11.9-slim

# Set working directory (container filesystem)
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first for layer caching
COPY requirements.txt ./

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Streamlit config (optional defaults)
ENV STREAMLIT_SERVER_PORT=8501

# Expose port
EXPOSE 8501

# Default command; rely on env vars for secrets
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
