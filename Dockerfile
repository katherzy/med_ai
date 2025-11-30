## Stage 1: build dependencies and install Python packages
#FROM python:3.12-slim AS builder
#WORKDIR /app
#
## System deps required for building wheels and for FAISS/numeric libs
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    git \
#    curl \
#    pkg-config \
#    libsndfile1 \
#    libopenblas-dev \
#    liblapack-dev \
#    && rm -rf /var/lib/apt/lists/*
#
## Install uv (fast dependency manager) into builder env
#RUN pip install --no-cache-dir uv
#
## Copy only lock files and pyproject for reproducible install
#COPY pyproject.toml uv.lock /app/
#
## Install dependencies into /install (site-packages) so we can copy them into final
#RUN uv sync --frozen
#
## Stage 2: runtime image
#FROM python:3.12-slim
#WORKDIR /app
#
## System runtime deps (keep minimal)
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libsndfile1 \
#    libstdc++6 \
#    && rm -rf /var/lib/apt/lists/*
#
##libopenblas-base \
#
## Copy Python packages from builder
#COPY --from=builder /install /usr/local/lib/python3.12/site-packages
#
## Create app user
#RUN useradd --create-home appuser
#USER appuser
#ENV HOME=/home/appuser
#WORKDIR /home/appuser/app
#
## Copy only application files (avoid copying .venv)
#COPY --chown=appuser:appuser pyproject.toml README.md chatbot.py create_vector_db.py utils.py ./
#COPY --chown=appuser:appuser data/ ./data
##COPY --chown=appuser:appuser vectorstore/ ./vectorstore
#
## Expose port your app uses (example for FastAPI/uvicorn)
#EXPOSE 8501
#
## Do not copy secrets; expect .env provided at runtime (docker run -v or --env-file)
#ENV PYTHONUNBUFFERED=1
#
## Default command - adjust to how you run your app (streamlit, uvicorn, etc.)
#
#CMD ["streamlit","run","chatbot.py","--server.port","8501","--server.address","0.0.0.0"]


FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl pkg-config cmake libsndfile1 libopenblas-dev liblapack-dev libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY chatbot.py create_vector_db.py utils.py README.md ./
COPY data/ ./data
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["/bin/bash"]
#CMD ["streamlit","run","chatbot.py","--server.port","8501","--server.address","0.0.0.0"]



# Multi-stage build for smaller image size
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    pkg-config \
    cmake \
    libsndfile1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy Python packages from builder to appuser's home with proper ownership
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

USER appuser
WORKDIR /home/appuser/app

# Copy application files
COPY --chown=appuser:appuser chatbot.py create_vector_db.py utils.py README.md ./
COPY --chown=appuser:appuser data/ ./data/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "chatbot.py", "--server.port", "8501", "--server.address", "0.0.0.0"]



# docker build -t med_ai_uv .
# docker run -it --rm -v $(pwd):/app med_ai_uv

#Part	Meaning
#-v	    Mounts a volume (directory)
#$(pwd)	Current folder on your host machine (local machine)
#/app	Folder inside the Docker container where the host folder will appear

# docker ps #view all containers
# python -c "import torch, numpy, sentence_transformers; print(torch.__version__, numpy.__version__)"
# exit #to stop from inside container
# docker stop $(docker ps -q)
# docker run -it --rm -p 8501:8501 med_ai_uv
# docker run -it --rm -v $(pwd):/app -p 8501:8501 med_ai_uv
# streamlit run chatbot.py --server.port 8501 --server.address 0.0.0.0
# curl -I https://huggingface.co