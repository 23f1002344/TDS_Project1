FROM python:3.12-slim-bookworm

# Install dependencies, including Node.js from NodeSource
# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    sqlite3 \
    ffmpeg \
    imagemagick \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install NodeJS
RUN curl -sL https://deb.nodesource.com/setup_22.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get install -y nodejs && \
    node -v && \
    npm install -g prettier@3.4.2

# Upgrade pip and install packages in system Python
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
    # Web frameworks and APIs
    fastapi \
    uvicorn[standard] \
    requests \
    httpx \
    aiohttp \
    websockets \
    # Data processing
    pandas \
    numpy \
    duckdb \
    polars \
    pyarrow \
    scipy \
    # Database
    sqlalchemy \
    psycopg2-binary \
    asyncpg \
    # ML and AI
    # scikit-learn \
    # transformers \
    # sentence-transformers \
    # torch \
    # Image processing
    pillow \
    opencv-python-headless \
    # Text processing
    beautifulsoup4 \
    markdown \
    python-frontmatter \
    pyyaml \
    jinja2 \
    # File handling
    python-multipart \
    python-magic \
    python-docx \
    pypdf2 \
    # Audio processing
    librosa \
    # Development tools
    pytest \
    black \
    isort \
    mypy \
    python-dotenv \
    rich \
    typer \
    # Git operations
    gitpython

# Install uv package manager
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx/bin/

# Create and set working directory
WORKDIR /app

# Create data directory with proper permissions
RUN mkdir -p /data

# # Set environment variables
# ARG AIPROXY_TOKEN
# ENV AIPROXY_TOKEN=${AIPROXY_TOKEN}

# Ignore pip complaining about root user
ENV PIP_ROOT_USER_ACTION=ignore

# Copy application files
COPY app.py .

# Expose port for FastAPI
EXPOSE 8000

# Run the application
CMD ["uv", "run", "app.py"]
