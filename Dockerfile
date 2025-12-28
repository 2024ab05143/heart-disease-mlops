# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Upgrade pip first to avoid issues
RUN pip install --upgrade pip

# Copy requirements and install dependencies
# We add --default-timeout=100 to prevent network timeouts
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the model and code
COPY model/ model/
COPY scaler.pkl .
COPY src/app.py .

# Expose port 5000
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]