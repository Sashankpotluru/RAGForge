FROM python:3.13.9-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/runtime && chown appuser /app/data/runtime
USER appuser
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "ragforge.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
