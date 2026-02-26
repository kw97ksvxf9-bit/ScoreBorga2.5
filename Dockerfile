# ScoreBorga 2.5 — Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Default command: start the scheduler (runs every Friday at PREDICTION_RUN_TIME)
# Override with --run-now to execute immediately:
#   docker run --env-file .env scoreborga python scheduler/weekend_runner.py --run-now
CMD ["python", "scheduler/weekend_runner.py"]
