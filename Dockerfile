FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# App code + trained model (run `python -m car_price_ml.train` to produce models/ first)
COPY api ./api
COPY models ./models
# Static valuation form served by the API at the site root
COPY docs/app ./docs/app

# Run as a non-root user (least privilege)
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
