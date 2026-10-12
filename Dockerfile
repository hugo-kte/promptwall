# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .
COPY configs ./configs

RUN useradd --create-home --uid 10001 promptwall && mkdir -p runtime artifacts && chown -R promptwall /app
USER promptwall

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
ENTRYPOINT ["promptwall"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--policy", "configs/policy.yaml"]
