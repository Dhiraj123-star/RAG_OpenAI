# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "rag:app", "--host", "0.0.0.0", "--port", "8000"]
