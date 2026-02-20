FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Install the project (non-editable)
COPY src src
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.13-slim

# Install libtorrent-rasterbar Python bindings (C++ extension, not pip-installable)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-libtorrent && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Symlink system libtorrent into the venv
RUN VENV_SITE=$(/app/.venv/bin/python -c "import site; print(site.getsitepackages()[0])") && \
    SYS_SITE=$(find /usr/lib/python3 -name "libtorrent*" -path "*/dist-packages/*" | head -1 | xargs dirname) && \
    if [ -n "$SYS_SITE" ]; then \
      ln -sf "$SYS_SITE"/libtorrent* "$VENV_SITE/"; \
    fi

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["annas-dl"]
CMD ["serve"]
