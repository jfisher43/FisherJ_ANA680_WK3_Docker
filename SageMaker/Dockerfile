# Start with a base image that supports multi-architecture builds
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables to prevent Python from generating .pyc files
ENV PYTHONDONTWRITEBYTECODE=TRUE
# Ensure that Python output is sent directly to the terminal
ENV PYTHONUNBUFFERED=TRUE

# Update PATH environment variable
ENV PATH="/app:${PATH}"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the pre-trained model and the Flask application script into the container
COPY model.joblib .
COPY serve.py .

# Expose the port the app runs on
EXPOSE 8080

# Use Gunicorn to serve the Flask application
# Adjust the module path to your Flask app's Python path
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8080", "serve:app"]