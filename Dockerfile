FROM python:3.12-slim AS trainer
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY titanic_mlops/ ./titanic_mlops/
COPY scripts/__init__.py scripts/prepare_deployment_model.py ./scripts/
RUN python -m scripts.prepare_deployment_model

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TITANIC_MODEL_PATH=/app/artifacts/survival-model/model.joblib
WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt
COPY titanic_mlops/ ./titanic_mlops/
COPY scripts/__init__.py scripts/serve_api.py ./scripts/
COPY --from=trainer /app/artifacts/survival-model/ ./artifacts/survival-model/
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app/artifacts
USER appuser
EXPOSE 8000
CMD ["python", "-m", "scripts.serve_api", "--host", "0.0.0.0", "--port", "8000"]
