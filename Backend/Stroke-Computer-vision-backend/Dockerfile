# Use official Python slim image
FROM python:3.12.4-slim

# Create and use non-root user later
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create user and switch context
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install Python packages
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy source code
COPY --chown=user . /app

# Start with Gunicorn (adjust port or app path if needed)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "main:app"]
