# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip wheel --wheel-dir /wheels ".[ml,live]"

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="quantbot" \
      org.opencontainers.image.description="Leakage-aware trading research and paper execution" \
      org.opencontainers.image.source="https://github.com/vinwiegman/quant"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels quantbot==0.1.0 \
    && rm -rf /wheels \
    && useradd --create-home --uid 10001 quantbot \
    && mkdir -p /workspace/results /workspace/state /workspace/logs \
    && chown -R quantbot:quantbot /workspace

USER quantbot
WORKDIR /workspace
VOLUME ["/workspace/results", "/workspace/state", "/workspace/logs"]

ENTRYPOINT ["quantbot"]
CMD ["--help"]
