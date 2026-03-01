#!/bin/bash

# Run code quality checks
set -e

echo "=== Formatting check (black) ==="
uv run black --check backend/

echo "=== Tests (pytest) ==="
cd backend && uv run pytest tests/ -v
