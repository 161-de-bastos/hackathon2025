FROM python:3.11-slim
WORKDIR /app
COPY fastapi/app /app
RUN pip install --no-cache-dir fastapi uvicorn requests
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
