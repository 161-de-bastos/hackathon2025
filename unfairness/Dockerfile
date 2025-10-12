FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
