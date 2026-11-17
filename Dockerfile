FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY models ./models

RUN pip install --no-cache-dir -e .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "pitchmodel.api:app", "--host", "0.0.0.0", "--port", "8000"]