# Red Team Agent — API konteyner imajı
FROM python:3.13-slim

# Loglar anında aksın, pip cache yazma.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Önce bağımlılıklar (katman cache'i — kod değişince pip tekrar çalışmaz).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Uygulama kodu.
COPY . .

# FastAPI/uvicorn portu.
EXPOSE 8000

# API'yi tüm arayüzlerde dinlet (konteyner dışından erişilebilsin).
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
