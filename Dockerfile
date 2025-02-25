FROM python:3.12-slim

# Install system dependencies needed for playwright and other libraries.
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    libnss3 \
    libx11-6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory.
WORKDIR /app

# Copy the requirements file and install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Install Playwright browsers (required by crawl4ai).
RUN playwright install

# Expose Streamlit's default port.
EXPOSE 8501

# Start the Streamlit app.
CMD ["streamlit", "run", "app.py", "--server.enableCORS", "false"]
