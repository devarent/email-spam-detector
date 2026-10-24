FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir numpy pandas scikit-learn matplotlib seaborn flask flask-cors

# Copy application files
COPY . .

EXPOSE 5000

# Run the spam detection experiment, then keep container alive serving on 5000
CMD ["sh", "-c", "python SpamMailPrediction.py; python -m http.server 5000"]
