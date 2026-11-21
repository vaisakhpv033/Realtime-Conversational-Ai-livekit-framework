#!/usr/bin/env bash
# LiveKit Voice Agent Framework - Automated Production Deployment Script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=================================================="
echo " LiveKit Voice Agent Framework - Deployment Rollout"
echo " Workspace: ${ROOT_DIR}"
echo "=================================================="

cd "${ROOT_DIR}"

# 1. Verify Docker daemon is running
echo "[1/5] Verifying Docker daemon status..."
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running or current user lacks Docker socket permissions." >&2
    exit 1
fi
echo "Docker daemon is active."

# 2. Verify .env file and required credentials
echo "[2/5] Validating environment configuration (.env)..."
if [ ! -f ".env" ]; then
    echo "ERROR: Missing .env file in ${ROOT_DIR}." >&2
    echo "Please copy .env.example to .env and configure production credentials." >&2
    exit 1
fi

REQUIRED_VARS=("LIVEKIT_URL" "LIVEKIT_API_KEY" "LIVEKIT_API_SECRET" "GOOGLE_API_KEY")
for var_name in "${REQUIRED_VARS[@]}"; do
    val=$(grep -E "^${var_name}=" .env | head -n1 | cut -d '=' -f2- | tr -d ' "\r' || true)
    if [ -z "${val}" ]; then
        echo "ERROR: Required environment variable '${var_name}' is missing or empty in .env" >&2
        exit 1
    fi
    if [[ "${val}" == your-* ]]; then
        echo "ERROR: Placeholder value detected for '${var_name}' in .env ('${val}'). Please configure valid credentials." >&2
        exit 1
    fi
done
echo "Required environment variables validated."

# 3. Build optimized multi-stage image
echo "[3/5] Building container image..."
docker compose build agent-framework

# 4. Zero-downtime container rollout
echo "[4/5] Deploying container stack..."
docker compose up -d --remove-orphans agent-framework

# 5. Health verification polling loop
HEALTH_PORT=$(grep -E "^HEALTH_CHECK_PORT=" .env | head -n1 | cut -d '=' -f2- | tr -d ' "\r' || true)
HEALTH_PORT="${HEALTH_PORT:-8081}"
HEALTH_URL="http://127.0.0.1:${HEALTH_PORT}/"
MAX_ATTEMPTS=15
SLEEP_SECONDS=3
HEALTHY=false

echo "[5/5] Polling health check probe at ${HEALTH_URL}..."
for i in $(seq 1 "${MAX_ATTEMPTS}"); do
    if curl -s -f -o /dev/null "${HEALTH_URL}"; then
        HEALTHY=true
        echo "Worker health probe succeeded on attempt ${i} (HTTP 200 OK)."
        break
    fi
    echo "Worker starting up... (attempt ${i}/${MAX_ATTEMPTS})"
    sleep "${SLEEP_SECONDS}"
done

if [ "${HEALTHY}" = false ]; then
    echo "ERROR: Worker health check failed after $((MAX_ATTEMPTS * SLEEP_SECONDS)) seconds." >&2
    echo "Recent container logs:" >&2
    docker compose logs --tail=60 agent-framework >&2
    exit 1
fi

echo "=================================================="
echo " Deployment completed successfully."
echo " Worker is healthy and ready to accept LiveKit jobs."
echo "=================================================="
