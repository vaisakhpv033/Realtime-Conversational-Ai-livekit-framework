"""Unit tests for the deploy/verify_stack.py utility."""

import importlib.util
import os
import sys
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Dynamically import verify_stack.py from deploy/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VERIFY_STACK_PATH = REPO_ROOT / "deploy" / "verify_stack.py"

spec = importlib.util.spec_from_file_location("verify_stack", VERIFY_STACK_PATH)
verify_stack = importlib.util.module_from_spec(spec)
sys.modules["verify_stack"] = verify_stack
spec.loader.exec_module(verify_stack)
StackVerifier = verify_stack.StackVerifier


def test_verify_env_detects_missing_keys():
    verifier = StackVerifier(skip_sfu=True, skip_gemini=True, skip_health=True)
    verifier.env_vars = {}
    success = verifier.verify_env_variables()
    assert success is False
    assert len(verifier.errors) == 4
    assert any("LIVEKIT_URL" in e for e in verifier.errors)
    assert any("LIVEKIT_API_KEY" in e for e in verifier.errors)
    assert any("LIVEKIT_API_SECRET" in e for e in verifier.errors)
    assert any("GOOGLE_API_KEY" in e for e in verifier.errors)


def test_verify_env_detects_placeholder_values():
    verifier = StackVerifier(skip_sfu=True, skip_gemini=True, skip_health=True)
    verifier.env_vars = {
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": "your-api-key",
        "LIVEKIT_API_SECRET": "your-secret",
        "GOOGLE_API_KEY": "your-gemini-key",
    }
    success = verifier.verify_env_variables()
    assert success is False
    assert len(verifier.errors) == 3
    assert all("Placeholder" in e for e in verifier.errors)


def test_verify_env_succeeds_with_valid_and_dev_credentials():
    verifier = StackVerifier(skip_sfu=True, skip_gemini=True, skip_health=True)
    verifier.env_vars = {
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "secret_livekit_key_for_local_testing",
        "GOOGLE_API_KEY": "AIzaSyValidProductionFormatKey12345",
        "DRAIN_TIMEOUT_SECONDS": "300",
        "WORKER_LOAD_THRESHOLD": "0.7",
        "LIVEKIT_LOG_LEVEL": "INFO",
    }
    success = verifier.verify_env_variables()
    assert success is True
    assert len(verifier.errors) == 0


@pytest.mark.asyncio
async def test_verify_health_server_success():
    verifier = StackVerifier(skip_sfu=True, skip_gemini=True, skip_health=False, health_port=8081)

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value="OK")

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_resp), __aexit__=AsyncMock()))

    with patch.object(verify_stack.aiohttp, "ClientSession", return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())):
        result = await verifier.verify_health_server()
        assert result is True


@pytest.mark.asyncio
async def test_verify_health_server_unreachable_is_non_fatal():
    verifier = StackVerifier(skip_sfu=True, skip_gemini=True, skip_health=False, health_port=59999)
    # When no server is listening on port 59999, probing logs notice and returns True (non-fatal)
    result = await verifier.verify_health_server()
    assert result is True
