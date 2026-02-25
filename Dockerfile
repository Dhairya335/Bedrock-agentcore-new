FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip uv

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen

COPY src /app/src

EXPOSE 8080

CMD ["python", "-m", "src.main"]
