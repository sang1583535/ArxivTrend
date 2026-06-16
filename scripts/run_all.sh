#!/usr/bin/env bash

set -e

BACKEND_HOST=${BACKEND_HOST:-0.0.0.0}
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_HOST=${FRONTEND_HOST:-0.0.0.0}
FRONTEND_PORT=${FRONTEND_PORT:-8501}

echo "Starting backend on ${BACKEND_HOST}:${BACKEND_PORT}..."
uvicorn backend.main:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
BACKEND_PID=$!

echo "Starting Streamlit frontend on ${FRONTEND_HOST}:${FRONTEND_PORT}..."
streamlit run frontend/app.py --server.address "${FRONTEND_HOST}" --server.port "${FRONTEND_PORT}" &
FRONTEND_PID=$!

cleanup() {
  echo "Stopping services..."
  kill "${BACKEND_PID}" "${FRONTEND_PID}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

wait