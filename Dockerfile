FROM python:3.11-bullseye

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    ffmpeg \
    google-perftools \
    ca-certificates curl gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the new user
USER user
WORKDIR /home/user/app

# Copy requirements first to leverage Docker cache
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user:user . .

# Add local bin to PATH to avoid warnings
ENV PATH="/home/user/.local/bin:${PATH}"

# Run the FastAPI application
CMD ["uvicorn", "food_ordering_livekit:app", "--host", "0.0.0.0", "--port", "8080"]
