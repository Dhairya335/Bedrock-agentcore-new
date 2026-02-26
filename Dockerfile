FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen

COPY src /app/src

EXPOSE 8080

CMD ["python", "-m", "src.main"]
