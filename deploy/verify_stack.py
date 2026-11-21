#!/usr/bin/env python3
"""Production Stack Verification Script for LiveKit Voice Agent Framework.

Performs offline and online pre-flight / post-deployment checks:
1. Validates required environment variables and configuration consistency.
2. Tests LiveKit SFU network reachability and WebSocket connectivity.
3. Validates Google Gemini 3.8 Live API key credentials.
4. Probes the local AgentServer HTTP health check endpoint on port 8081.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp
from dotenv import dotenv_values

# Locate root directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class StackVerifier:
    """Verifies infrastructure and operational readiness of the Agent Framework."""

    def __init__(
        self,
        env_path: Path = DEFAULT_ENV_FILE,
        skip_sfu: bool = False,
        skip_gemini: bool = False,
        skip_health: bool = False,
        health_port: Optional[int] = None,
    ) -> None:
        self.env_path = env_path
        self.skip_sfu = skip_sfu
        self.skip_gemini = skip_gemini
        self.skip_health = skip_health
        self.health_port_override = health_port
        self.env_vars: dict[str, str] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def load_environment(self) -> None:
        """Loads environment variables from file with OS environment overrides."""
        file_vars: dict[str, Any] = {}
        if self.env_path.exists():
            file_vars = dotenv_values(self.env_path)
            print(f"[OK] Loaded environment template from: {self.env_path}")
        else:
            print(f"[WARN] No .env file found at: {self.env_path}. Relying on system environment.")

        # Combine with active OS environment (OS env takes precedence)
        combined: dict[str, str] = {}
        for k, v in file_vars.items():
            if v is not None:
                combined[k] = str(v).strip()
        for k, v in os.environ.items():
            combined[k] = str(v).strip()

        self.env_vars = combined

    def verify_env_variables(self) -> bool:
        """Validates the presence of required and optional environment keys."""
        print("\n--- 1. Checking Environment Variables ---")
        required = [
            ("LIVEKIT_URL", "LiveKit SFU endpoint (ws:// or wss://)"),
            ("LIVEKIT_API_KEY", "LiveKit project API key"),
            ("LIVEKIT_API_SECRET", "LiveKit project API secret"),
            ("GOOGLE_API_KEY", "Google Gemini API key"),
        ]

        success = True
        for key, description in required:
            val = self.env_vars.get(key, "").strip()
            if not val:
                self.errors.append(f"Missing required environment variable: {key} ({description})")
                print(f"  [FAIL] {key}: NOT SET")
                success = False
            elif val.startswith("your-"):
                self.errors.append(f"Placeholder value detected for {key}: '{val}'")
                print(f"  [FAIL] {key}: PLACEHOLDER DETECTED ('{val}')")
                success = False
            else:
                masked = f"{val[:4]}...{val[-4:]}" if len(val) > 8 else "***"
                if val in ("devkey", "secret_livekit_key_for_local_testing"):
                    print(f"  [OK]   {key}: {masked} (local development credential)")
                else:
                    print(f"  [OK]   {key}: {masked}")

        # Check production runtime variables
        drain = self.env_vars.get("DRAIN_TIMEOUT_SECONDS", "300.0")
        load_thr = self.env_vars.get("WORKER_LOAD_THRESHOLD", "0.7")
        log_lvl = self.env_vars.get("LIVEKIT_LOG_LEVEL", "INFO")
        prom_port = self.env_vars.get("PROMETHEUS_PORT", "disabled")

        print(f"  [INFO] DRAIN_TIMEOUT_SECONDS: {drain}")
        print(f"  [INFO] WORKER_LOAD_THRESHOLD: {load_thr}")
        print(f"  [INFO] LIVEKIT_LOG_LEVEL: {log_lvl}")
        print(f"  [INFO] PROMETHEUS_PORT: {prom_port}")

        return success

    async def verify_livekit_sfu(self) -> bool:
        """Validates network reachability and handshake to LiveKit SFU."""
        print("\n--- 2. Verifying LiveKit SFU Connectivity ---")
        if self.skip_sfu:
            print("  [SKIP] LiveKit SFU connectivity check skipped by user flag.")
            return True

        lk_url = self.env_vars.get("LIVEKIT_URL", "").strip()
        if not lk_url:
            self.errors.append("LIVEKIT_URL is not configured; cannot test SFU connectivity.")
            print("  [FAIL] LIVEKIT_URL is not set.")
            return False

        # LiveKit URLs are typically ws:// or wss://
        # We attempt a WebSocket handshake with a 5-second timeout
        timeout = aiohttp.ClientTimeout(total=5.0)
        start_time = time.perf_counter()

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.ws_connect(lk_url) as ws:
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        print(f"  [OK] Connected to LiveKit SFU at {lk_url} (latency: {latency_ms:.1f}ms)")
                        await ws.close()
                        return True
                except aiohttp.WSServerHandshakeError as e:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    # HTTP 401 Unauthorized or 400 Bad Request indicates SFU is alive and enforcing auth
                    if e.status in (400, 401, 403):
                        print(
                            f"  [OK] LiveKit SFU reachable at {lk_url} "
                            f"(HTTP {e.status} response from server, latency: {latency_ms:.1f}ms)"
                        )
                        return True
                    self.errors.append(f"LiveKit SFU handshake failed with HTTP {e.status}: {e.message}")
                    print(f"  [FAIL] LiveKit SFU handshake failed (HTTP {e.status}): {e.message}")
                    return False
        except asyncio.TimeoutError:
            self.errors.append(f"Connection to LiveKit SFU timed out after 5.0s: {lk_url}")
            print(f"  [FAIL] Connection to {lk_url} timed out (5.0s).")
            return False
        except Exception as e:
            self.errors.append(f"LiveKit SFU connectivity error ({type(e).__name__}): {e}")
            print(f"  [FAIL] Cannot connect to LiveKit SFU at {lk_url}: {e}")
            return False

    async def verify_gemini_api(self) -> bool:
        """Validates Google Gemini API key by making a lightweight discovery call."""
        print("\n--- 3. Verifying Gemini API Key Credentials ---")
        if self.skip_gemini:
            print("  [SKIP] Gemini API verification skipped by user flag.")
            return True

        api_key = self.env_vars.get("GOOGLE_API_KEY", "").strip()
        if not api_key or api_key.startswith("your-"):
            self.warnings.append("Valid GOOGLE_API_KEY not found; skipping Gemini API verification.")
            print("  [WARN] Valid GOOGLE_API_KEY not found; skipping API check.")
            return True

        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            # Query models with limit 1 to confirm credential validity
            models = list(client.models.list(config={"page_size": 1}))
            if models:
                print(f"  [OK] Gemini API credentials valid. Available model: {models[0].name}")
            else:
                print("  [OK] Gemini API credentials valid (empty model list returned).")
            return True
        except Exception as e:
            err_msg = str(e)
            if "Forbidden" in err_msg or "403" in err_msg or "API key not valid" in err_msg:
                self.errors.append(f"Invalid Gemini API key: {err_msg}")
                print(f"  [FAIL] Gemini API key rejected: {err_msg}")
                return False
            # Network error or non-auth issue can be a warning in offline verification
            self.warnings.append(f"Gemini API check encountered non-fatal error: {err_msg}")
            print(f"  [WARN] Gemini API connectivity check issue: {err_msg}")
            return True

    async def verify_health_server(self) -> bool:
        """Probes the AgentServer HTTP health check endpoint on port 8081."""
        print("\n--- 4. Probing Local Health Server Endpoint ---")
        if self.skip_health:
            print("  [SKIP] Health server probe skipped by user flag.")
            return True

        port = self.health_port_override
        if port is None:
            raw_port = self.env_vars.get("HEALTH_CHECK_PORT", "8081")
            try:
                port = int(raw_port)
            except ValueError:
                port = 8081

        health_url = f"http://127.0.0.1:{port}/"
        timeout = aiohttp.ClientTimeout(total=3.0)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        print(f"  [OK] Health server responding at {health_url} (HTTP 200: {text.strip() or 'OK'})")
                        return True
                    else:
                        self.warnings.append(f"Health server returned non-200 code: HTTP {resp.status}")
                        print(f"  [WARN] Health server returned HTTP {resp.status} at {health_url}")
                        return False
        except Exception as e:
            # If the container isn't running yet, this is a notice rather than fatal error
            print(f"  [INFO] Health server not responding on {health_url} ({type(e).__name__}).")
            print("         (Normal if the container/worker is not currently running locally).")
            return True

    async def run_all(self) -> int:
        """Runs the entire verification suite."""
        self.load_environment()

        env_ok = self.verify_env_variables()
        sfu_ok = await self.verify_livekit_sfu()
        gemini_ok = await self.verify_gemini_api()
        health_ok = await self.verify_health_server()

        print("\n================ Verification Summary ================")
        if self.errors:
            print(f"FAILED: Encountered {len(self.errors)} error(s):")
            for err in self.errors:
                print(f"  - {err}")
        else:
            print("SUCCESS: All mandatory verification checks passed.")

        if self.warnings:
            print(f"\nNotices / Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                print(f"  - {warn}")

        print("======================================================")
        return 1 if self.errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify operational readiness of the LiveKit Agent Framework stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to environment file to validate.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override health check port (default reads HEALTH_CHECK_PORT or 8081).",
    )
    parser.add_argument(
        "--skip-sfu",
        action="store_true",
        help="Skip LiveKit SFU WebSocket connectivity check.",
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Skip Google Gemini API credential check.",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip local HTTP health server probe.",
    )

    args = parser.parse_args()

    verifier = StackVerifier(
        env_path=args.env_file,
        skip_sfu=args.skip_sfu,
        skip_gemini=args.skip_gemini,
        skip_health=args.skip_health,
        health_port=args.port,
    )

    exit_code = asyncio.run(verifier.run_all())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
