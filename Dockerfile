# Streamlit dashboard image. Serves dashboard/app.py with the committed models.
# NOTE: written 2026-11-11; build/startup status is recorded in README
# ("Docker" section) and must be re-verified wherever Docker is available.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA wheel.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake the MiniLM weights into the image so the container starts offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY config.yaml ./
COPY .streamlit ./.streamlit
COPY src ./src
COPY dashboard ./dashboard
COPY models ./models
COPY data/processed/dashboard_categories.json ./data/processed/dashboard_categories.json
COPY data/labeled/hand_labeled.csv ./data/labeled/hand_labeled.csv

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
