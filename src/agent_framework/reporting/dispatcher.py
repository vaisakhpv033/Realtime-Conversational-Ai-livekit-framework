"""Asynchronous reporting webhook dispatcher with SSRF guards and retry resilience.

Features:
- Asynchronous HTTP dispatch using aiohttp.ClientSession
- Supports HTTP POST and PUT methods
- Authentication support: Bearer tokens, API keys, and HMAC-SHA256 request signing
- Pre-dispatch SSRF check before each request / retry
- Exponential backoff retry loop with jitter for transient errors (5xx, 429, timeouts, connection drops)
- Fail-fast on non-retryable 4xx client errors
- Local dead-letter JSON logging on exhausted retries or security violations
- Guaranteed exception safety: returns structured DeliveryResult without throwing unhandled exceptions
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import hmac
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Mapping, Optional, Union
import urllib.parse
import uuid
import aiohttp
from typing_extensions import Self

from agent_framework.config.schema import ReportingAuthConfig, ReportingConfig, SessionConfig
from agent_framework.reporting.payload_builder import build_session_telemetry_payload
from agent_framework.reporting.security import is_safe_webhook_url
from agent_framework.utils.secrets import resolve_secret

logger = logging.getLogger("agent_framework.reporting.dispatcher")


@dataclass
class DeliveryResult:
    """Structured result returned by the reporting dispatcher."""

    success: bool
    status_code: Optional[int] = None
    error: Optional[str] = None
    attempts: int = 0
    dead_letter_path: Optional[str] = None
    response_body: Optional[str] = None


def generate_auth_headers(
    auth: ReportingAuthConfig | None,
    payload_bytes: bytes,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Generates HTTP authentication headers for the webhook request.

    Supports:
    - 'bearer': Sets 'Authorization: Bearer <token>'
    - 'api_key': Sets 'X-API-Key: <key>'
    - 'hmac_sha256': Computes HMAC-SHA256 signature of the payload body,
                     setting 'X-Signature-SHA256: sha256=<sig>' and 'X-Signature: <sig>'

    Args:
        auth: Webhook auth configuration.
        payload_bytes: Serialized JSON payload bytes.
        env: Optional environment mapping for secret resolution.

    Returns:
        Dictionary of headers to inject.
    """
    if not auth or not auth.type or not auth.secret_ref:
        return {}

    secret = resolve_secret(auth.secret_ref, env=env, strict=True)
    auth_type = auth.type.lower().strip()
    headers: dict[str, str] = {}

    if auth_type == "bearer":
        token = secret if secret.startswith("Bearer ") else f"Bearer {secret}"
        headers["Authorization"] = token
    elif auth_type == "api_key":
        headers["X-API-Key"] = secret
    elif auth_type == "hmac_sha256":
        signature = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        headers["X-Signature-SHA256"] = f"sha256={signature}"
        headers["X-Signature"] = signature
    else:
        raise ValueError(f"Unsupported reporting auth type: {auth.type}")

    return headers


class ReportingDispatcher:
    """Asynchronous HTTP client for resilient post-session telemetry delivery."""

    def __init__(
        self,
        config: Union[ReportingConfig, SessionConfig, None] = None,
        session: Optional[aiohttp.ClientSession] = None,
        dead_letter_dir: Union[str, Path] = "logs/dead_letter",
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Initializes the reporting dispatcher.

        Args:
            config: ReportingConfig or SessionConfig. If None, uses default ReportingConfig.
            session: Optional pre-configured aiohttp.ClientSession.
            dead_letter_dir: Directory where undeliverable reports are saved.
            env: Optional environment mapping for secret resolution.
        """
        if isinstance(config, SessionConfig):
            self.config = config.reporting
        elif isinstance(config, ReportingConfig):
            self.config = config
        else:
            self.config = ReportingConfig()

        self._session = session
        self._owns_session = False
        self.dead_letter_dir = Path(dead_letter_dir)
        self.env = env

    async def __aenter__(self) -> Self:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Closes the HTTP session if owned by this dispatcher."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _write_dead_letter(
        self,
        payload: dict[str, Any],
        target_url: str,
        method: str,
        error_reason: str,
        attempts: int,
        status_code: Optional[int] = None,
    ) -> Optional[Path]:
        """Serializes undeliverable payload and error diagnostics to local disk."""
        try:
            self.dead_letter_dir.mkdir(parents=True, exist_ok=True)

            session_id = str(payload.get("session_id") or payload.get("job_id") or "report")
            sanitized_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)[:64] or "report"
            timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            suffix = uuid.uuid4().hex[:8]
            filename = f"dead_letter_{sanitized_id}_{timestamp_str}_{suffix}.json"
            file_path = self.dead_letter_dir / filename

            dead_letter_record = {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "target_url": target_url,
                "method": method,
                "error": {
                    "reason": error_reason,
                    "status_code": status_code,
                },
                "attempts": attempts,
                "payload": payload,
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(dead_letter_record, f, indent=2, default=str)

            logger.warning(f"Undeliverable report written to dead-letter storage: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed writing dead-letter record to {self.dead_letter_dir}: {e}", exc_info=True)
            return None

    async def dispatch(
        self,
        payload: dict[str, Any],
        url_override: Optional[str] = None,
    ) -> DeliveryResult:
        """Dispatches a telemetry payload with retries, SSRF validation, and dead-letter fallback.

        Args:
            payload: Standardized session telemetry payload dictionary.
            url_override: Optional URL overriding the configured endpoint.

        Returns:
            DeliveryResult indicating success, attempts, and dead-letter path if failed.
        """
        # Determine target endpoint
        target_url = url_override or (self.config.endpoint.url if self.config.endpoint else None)

        if not self.config.enabled and not url_override:
            logger.info("Post-session reporting is disabled. Skipping dispatch.")
            return DeliveryResult(success=True, error="Reporting disabled; skipped", attempts=0)

        method = (self.config.endpoint.method if self.config.endpoint else "POST").upper()
        if method not in ("POST", "PUT"):
            method = "POST"

        if not target_url:
            error_msg = "No reporting endpoint URL configured."
            logger.error(error_msg)
            dl_path = self._write_dead_letter(payload, "", method, error_msg, 0, None)
            return DeliveryResult(
                success=False,
                error=error_msg,
                attempts=0,
                dead_letter_path=str(dl_path) if dl_path else None,
            )

        # Serialize payload
        try:
            payload_json = json.dumps(payload, default=str)
            payload_bytes = payload_json.encode("utf-8")
        except Exception as e:
            error_msg = f"Failed serializing reporting payload to JSON: {e}"
            logger.error(error_msg)
            dl_path = self._write_dead_letter(payload, target_url, method, error_msg, 0, None)
            return DeliveryResult(
                success=False,
                error=error_msg,
                attempts=0,
                dead_letter_path=str(dl_path) if dl_path else None,
            )

        # Prepare request headers
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "AgentFramework-Reporting/1.0",
        }

        # Merge custom configured headers
        if self.config.endpoint and self.config.endpoint.headers:
            for k, v in self.config.endpoint.headers.items():
                try:
                    headers[k] = resolve_secret(v, env=self.env, strict=True)
                except Exception as e:
                    error_msg = f"Failed resolving secret in header '{k}': {e}"
                    logger.error(error_msg)
                    dl_path = self._write_dead_letter(payload, target_url, method, error_msg, 0, None)
                    return DeliveryResult(
                        success=False,
                        error=error_msg,
                        attempts=0,
                        dead_letter_path=str(dl_path) if dl_path else None,
                    )

        # Inject authentication headers
        if self.config.auth:
            try:
                auth_headers = generate_auth_headers(self.config.auth, payload_bytes, env=self.env)
                headers.update(auth_headers)
            except Exception as e:
                error_msg = f"Failed generating webhook auth headers: {e}"
                logger.error(error_msg)
                dl_path = self._write_dead_letter(payload, target_url, method, error_msg, 0, None)
                return DeliveryResult(
                    success=False,
                    error=error_msg,
                    attempts=0,
                    dead_letter_path=str(dl_path) if dl_path else None,
                )

        # Retry configuration
        retry_cfg = self.config.retry_policy
        max_retries = retry_cfg.max_retries
        total_attempts = max_retries + 1
        current_delay = retry_cfg.initial_delay_ms / 1000.0
        backoff_factor = retry_cfg.backoff_factor
        timeout_seconds = self.config.endpoint.timeout_seconds if self.config.endpoint else 10.0

        # Acquire HTTP session
        owns_local_session = False
        client_session = self._session
        if client_session is None or client_session.closed:
            client_session = aiohttp.ClientSession()
            owns_local_session = True

        last_error: Optional[str] = None
        last_status: Optional[int] = None
        attempts_made = 0

        try:
            for attempt in range(1, total_attempts + 1):
                attempts_made = attempt

                # Pre-dispatch SSRF check before each request / retry (run off-thread to prevent event loop stalls)
                if self.config.security.validate_ssrf:
                    is_safe, ssrf_reason = await asyncio.to_thread(
                        is_safe_webhook_url,
                        target_url,
                        self.config.security.allow_local,
                        self.config.security.allowed_domains,
                    )
                    if not is_safe:
                        last_error = f"SSRF security check rejected URL: {ssrf_reason}"
                        logger.error(f"Reporting dispatch aborted: {last_error}")
                        # Immediately fail without retry for security rejections
                        break

                # Dispatch HTTP request with safe redirect validation
                current_request_url = target_url
                redirect_hops = 0
                max_redirects = 3
                attempt_succeeded = False
                should_abort_attempt = False

                while redirect_hops <= max_redirects:
                    try:
                        logger.debug(
                            f"Dispatching report via {method} to {current_request_url} "
                            f"(attempt {attempt}/{total_attempts}, hop {redirect_hops})"
                        )
                        async with client_session.request(
                            method=method,
                            url=current_request_url,
                            data=payload_bytes,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                            allow_redirects=False,
                        ) as resp:
                            last_status = resp.status
                            resp_text = await resp.text()

                            # Handle HTTP redirects safely (re-validating target URL against SSRF)
                            if resp.status in (301, 302, 303, 307, 308):
                                location = resp.headers.get("Location")
                                if not location:
                                    last_error = f"HTTP {resp.status} redirect without Location header"
                                    should_abort_attempt = True
                                    break

                                redirect_target = urllib.parse.urljoin(current_request_url, location)
                                if self.config.security.validate_ssrf:
                                    is_safe, redirect_reason = await asyncio.to_thread(
                                        is_safe_webhook_url,
                                        redirect_target,
                                        self.config.security.allow_local,
                                        self.config.security.allowed_domains,
                                    )
                                    if not is_safe:
                                        last_error = f"SSRF security check rejected redirect URL: {redirect_reason}"
                                        logger.error(f"Reporting redirect aborted: {last_error}")
                                        dl_path = self._write_dead_letter(
                                            payload=payload,
                                            target_url=redirect_target,
                                            method=method,
                                            error_reason=last_error,
                                            attempts=attempt,
                                            status_code=resp.status,
                                        )
                                        return DeliveryResult(
                                            success=False,
                                            status_code=resp.status,
                                            error=last_error,
                                            attempts=attempt,
                                            dead_letter_path=str(dl_path) if dl_path else None,
                                        )

                                redirect_hops += 1
                                if redirect_hops > max_redirects:
                                    last_error = f"Exceeded maximum redirect hops ({max_redirects})"
                                    should_abort_attempt = True
                                    break

                                logger.debug(f"Following safe redirect to: {redirect_target}")
                                current_request_url = redirect_target
                                continue

                            if 200 <= resp.status < 300:
                                logger.info(
                                    f"Successfully delivered session report to {current_request_url} on attempt {attempt}"
                                )
                                return DeliveryResult(
                                    success=True,
                                    status_code=resp.status,
                                    attempts=attempt,
                                    response_body=resp_text,
                                )

                            last_error = f"HTTP {resp.status}: {resp_text[:200]}"
                            logger.warning(
                                f"Reporting endpoint responded with error {resp.status} "
                                f"(attempt {attempt}/{total_attempts}): {last_error}"
                            )

                            # Non-retryable client errors (4xx except 429 rate limits)
                            if resp.status < 500 and resp.status != 429:
                                should_abort_attempt = True
                                break
                            break

                    except (TimeoutError, asyncio.TimeoutError):
                        last_error = f"Request timed out after {timeout_seconds}s"
                        logger.warning(f"Reporting timeout (attempt {attempt}/{total_attempts}): {last_error}")
                        break
                    except aiohttp.ClientError as e:
                        last_error = f"Network client error: {e}"
                        logger.warning(f"Reporting network error (attempt {attempt}/{total_attempts}): {last_error}")
                        break
                    except Exception as e:
                        last_error = f"Unexpected error during dispatch: {e}"
                        logger.error(
                            f"Unexpected dispatch error (attempt {attempt}/{total_attempts}): {last_error}",
                            exc_info=True,
                        )
                        should_abort_attempt = True
                        break

                if should_abort_attempt:
                    break

                # Sleep before retry if more attempts remain
                if attempt < total_attempts:
                    current_delay = min(current_delay, 60.0)
                    jitter = random.uniform(0.0, 0.1 * current_delay)
                    sleep_duration = current_delay + jitter
                    logger.debug(f"Retrying webhook dispatch in {sleep_duration:.2f}s...")
                    await asyncio.sleep(sleep_duration)
                    current_delay *= backoff_factor

        except Exception as e:
            last_error = f"Unhandled exception in dispatch loop: {e}"
            logger.error(last_error, exc_info=True)
        finally:
            if owns_local_session and client_session and not client_session.closed:
                await client_session.close()

        # All attempts failed or dispatch was aborted
        logger.error(f"Failed delivering session report after {attempts_made} attempts: {last_error}")
        dl_path = self._write_dead_letter(
            payload=payload,
            target_url=target_url,
            method=method,
            error_reason=last_error or "Unknown dispatch error",
            attempts=attempts_made,
            status_code=last_status,
        )

        return DeliveryResult(
            success=False,
            status_code=last_status,
            error=last_error,
            attempts=attempts_made,
            dead_letter_path=str(dl_path) if dl_path else None,
        )


async def send_session_report(
    ctx: Any,
    session_config: SessionConfig,
    lk_report: Any = None,
    error: Exception | None = None,
    termination_reason: str | None = None,
    session: Optional[aiohttp.ClientSession] = None,
    dead_letter_dir: Union[str, Path] = "logs/dead_letter",
    env: Optional[Mapping[str, str]] = None,
) -> DeliveryResult:
    """Convenience helper to assemble and dispatch session report telemetry.

    Args:
        ctx: LiveKit JobContext or mock context.
        session_config: Validated SessionConfig metadata contract.
        lk_report: Optional LiveKit SessionReport.
        error: Optional Exception causing termination.
        termination_reason: Optional explicit termination reason.
        session: Optional aiohttp.ClientSession.
        dead_letter_dir: Directory for dead-letter storage.
        env: Optional environment mapping for secret resolution.

    Returns:
        DeliveryResult describing outcome.
    """
    if not session_config.reporting.enabled:
        return DeliveryResult(success=True, error="Reporting disabled; skipped", attempts=0)

    payload = build_session_telemetry_payload(
        ctx=ctx,
        session_config=session_config,
        lk_report=lk_report,
        error=error,
        termination_reason=termination_reason,
    )

    dispatcher = ReportingDispatcher(
        config=session_config.reporting,
        session=session,
        dead_letter_dir=dead_letter_dir,
        env=env,
    )

    return await dispatcher.dispatch(payload)


# Bind send_session_report as a static method on ReportingDispatcher for unified invocation
ReportingDispatcher.send_session_report = staticmethod(send_session_report)
