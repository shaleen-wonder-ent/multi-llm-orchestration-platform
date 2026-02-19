# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files including .env
COPY . .

# Expose port 8000 (Azure will map this to PORT env variable)
EXPOSE 8000

# Use proper JSON array format for CMD to fix exit code 126
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]