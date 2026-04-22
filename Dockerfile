# Stage 1: Builder
# This stage installs all dependencies, including build tools, into a virtual environment.
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential

# Set the working directory
WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment for subsequent RUN commands
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file and install dependencies except torch
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > requirements-no-torch.txt && \
	pip install --no-cache-dir -r requirements-no-torch.txt && \
	pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch


# Stage 2: Final image
# This stage copies the application code and the virtual environment from the builder stage.
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the rest of the application code
COPY . .

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "flaskr:create_app()"]
