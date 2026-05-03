# LucidCoder — Terminal-Bench 2 coding agent (purple agent for AgentBeats)
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd -m -s /bin/bash agent
WORKDIR /home/agent/workspace

# Dependencies first (layer caching)
COPY workspace/requirements.txt ./
RUN pip install -r requirements.txt

# Source
COPY workspace/src/ ./src/

ENV SERVER_HOST=0.0.0.0 \
    SERVER_PORT=9009 \
    MODEL_ID=gemini-3.1-pro-preview \
    LOG_LEVEL=INFO

RUN mkdir -p /home/agent/workspace/scratch \
    && chown -R agent:agent /home/agent/workspace
USER agent

EXPOSE 9009

ENTRYPOINT ["python", "-m", "src.server"]
