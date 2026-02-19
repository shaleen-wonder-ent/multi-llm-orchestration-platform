# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Set environment variable for port (with fallback to 8000)
ENV PORT=8000

# Run the application using the PORT environment variable
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT