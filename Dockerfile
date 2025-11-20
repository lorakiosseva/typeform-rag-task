# Use a modern Python inside the container
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy project files into the image
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start the API with Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]