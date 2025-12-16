#!/usr/bin/env bash
echo "[env] Python / uv"

if ! command -v uv >/dev/null 2>&1; then
  echo "[env] Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
else
  echo "[env] Using existing uv installation"
fi

uv sync
