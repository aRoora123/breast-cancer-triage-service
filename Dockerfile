
FROM python:3.12-slim AS trainer

WORKDIR /build

ARG MODEL_VERSION=1.0.0

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY params.yaml .
COPY training/ training/
COPY data/raw/breast_cancer.csv data/raw/breast_cancer.csv

RUN python training/train.py \
    --data-path data/raw/breast_cancer.csv \
    --output-dir artifacts \
    --model-version "${MODEL_VERSION}"

FROM python:3.12-slim AS production

WORKDIR /app

ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.joblib \
    DECISION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    MODEL_VERSION=${MODEL_VERSION} \
    GIT_SHA=${GIT_SHA}

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --from=trainer /build/artifacts/ ./artifacts/
COPY app/ ./app/

LABEL org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}"

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
