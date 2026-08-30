FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

# node + npm so npx-based MCP servers work; uv provides uvx for python MCP servers
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm tzdata git \
    && rm -rf /var/lib/apt/lists/*

# Playwright MCP + a real headless Chromium so agents can browse the web.
# `playwright install --with-deps` pulls the system libraries; the second install
# fetches the exact browser build @playwright/mcp's bundled playwright expects.
RUN npm install -g playwright @playwright/mcp \
    && playwright install --with-deps chromium \
    && playwright-mcp install-browser chrome-for-testing \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uv

COPY diana ./diana

EXPOSE 8080 8443
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/api/health',timeout=4)" || exit 1

CMD ["python", "-m", "diana.run"]
