# ScoreBorga 2.5 — Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Ensure all top-level packages are importable
ENV PYTHONPATH=/app

# Install dependencies first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Default command: start the timezone-aware scheduler (runs every Friday at PREDICTION_RUN_TIME in TIMEZONE).
# Override at runtime:
#   Run once immediately:
#     docker run --env-file .env scoreborga python scheduler/weekend_runner.py --run-now
#   Run immediately then continue scheduling weekly:
#     docker run --env-file .env scoreborga python scheduler/weekend_runner.py --run-once-then-schedule
CMD ["python", "scheduler/weekend_runner.py"]
