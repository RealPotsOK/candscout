FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY live_sim/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

ENV HOME=/app/state \
    USER=candscout \
    LOGNAME=candscout \
    XDG_CACHE_HOME=/app/state/.cache \
    TORCH_HOME=/app/state/.cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/app/state/.cache/torchinductor \
    TRITON_CACHE_DIR=/app/state/.cache/triton \
    TORCHDYNAMO_DISABLE=1

COPY live_sim/app /app/app
COPY src /app/src

EXPOSE 8080
CMD ["python", "-m", "app.main"]
