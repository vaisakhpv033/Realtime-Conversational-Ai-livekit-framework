"""Unit tests for Phase 2: Resilient Reporting & SSRF Hardening.

Covers:
1. SSRF and DNS rebinding prevention (security.py)
2. Application-agnostic session telemetry payload building (payload_builder.py)
3. Authentication header generation (Bearer, API Key, HMAC-SHA256)
4. Dispatcher retries, exponential backoff, jitter, and eventual success (dispatcher.py)
5. Dispatcher dead-letter fallback on consistent failures without raising unhandled exceptions
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
import socket
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from agent_framework.config.schema import (
    AgentConfig,
    InstructionsConfig,
    ModelsConfig,
    ReportingAuthConfig,
    ReportingConfig,
    ReportingEndpointConfig,
    ReportingPayloadConfig,
    ReportingSecurityConfig,
    RetryPolicyConfig,
    SessionConfig,
    VoiceConfig,
)
from agent_framework.reporting.dispatcher import (
    DeliveryResult,
    ReportingDispatcher,
    generate_auth_headers,
    send_session_report,
)
from agent_framework.reporting.payload_builder import (
    build_session_telemetry_payload,
)
from agent_framework.reporting.security import (
    is_safe_webhook_url,
    validate_ip_address,
)
from agent_framework.utils.secrets import SecretResolutionError


# ==============================================================================
# Helper Fixtures & Mocks
# ==============================================================================
class MockHttpResponse:
    """Mock asynchronous response for aiohttp request context manager."""

    def __init__(
        self,
        status: int = 200,
        text: str = '{"status": "ok"}',
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.status = status
        self._text = text
        self.headers = headers or {}

    async def text(self) -> str:
        return self._text

    async def json(self) -> Any:
        return json.loads(self._text)

    async def __aenter__(self) -> MockHttpResponse:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class MockHttpSession:
    """Mock aiohttp.ClientSession tracking calls and providing programmed responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.call_count += 1
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        if not self.responses:
            return MockHttpResponse(200, "OK")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def minimal_session_config() -> SessionConfig:
    """Provides a valid minimal SessionConfig for reporting tests."""
    return SessionConfig(
        agent=AgentConfig(name="support-agent", role="customer_support"),
        models=ModelsConfig(provider="gemini", model="gemini-3.8-live"),
        instructions=InstructionsConfig(system_prompt="You are a helpful customer support agent."),
        voice=VoiceConfig(provider="gemini", voice_id="Aoede"),
        reporting=ReportingConfig(
            enabled=True,
            endpoint=ReportingEndpointConfig(
                url="https://api.example.com/reports",
                method="POST",
                timeout_seconds=5.0,
            ),
            retry_policy=RetryPolicyConfig(
                max_retries=2,
                initial_delay_ms=10,  # fast test execution
                backoff_factor=1.5,
            ),
            security=ReportingSecurityConfig(
                validate_ssrf=True,
                allow_local=False,
            ),
            payload_config=ReportingPayloadConfig(
                include_chat_history=True,
                include_model_usage=True,
                include_events=True,
                custom_fields={"tenant": "acme_corp", "tier": "gold"},
            ),
        ),
        metadata={"session_id": "test-sess-1234", "account_id": "acc-99"},
    )


# ==============================================================================
# 1. SSRF & Security Validation Tests
# ==============================================================================
class TestReportingSecurity:
    """Tests for SSRF prevention, IP filtering, and DNS rebinding."""

    def test_is_safe_webhook_url_valid_https(self) -> None:
        """Legitimate public HTTPS target should be permitted."""
        mock_addr = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        with patch("socket.getaddrinfo", return_value=mock_addr):
            safe, err = is_safe_webhook_url("https://api.example.com/reports")
            assert safe is True
            assert err is None

    def test_is_safe_webhook_url_blocks_cloud_metadata_literal(self) -> None:
        """Cloud metadata IP literals (169.254.169.254, 100.100.100.200) must be strictly blocked."""
        safe, err = is_safe_webhook_url("http://169.254.169.254/latest/meta-data/")
        assert safe is False
        assert "forbidden" in err.lower() or "cloud metadata" in err.lower()

        safe, err = is_safe_webhook_url("http://100.100.100.200/latest/meta-data/")
        assert safe is False
        assert "forbidden" in err.lower() or "cloud metadata" in err.lower()

    def test_is_safe_webhook_url_blocks_cloud_metadata_even_if_allow_local(self) -> None:
        """Cloud metadata must never be allowed even when allow_local=True."""
        safe, err = is_safe_webhook_url("http://169.254.169.254/latest/meta-data/", allow_local=True)
        assert safe is False
        assert "cloud metadata" in err.lower() or "forbidden" in err.lower()

    def test_is_safe_webhook_url_blocks_cloud_metadata_hostnames(self) -> None:
        """Known cloud metadata hostnames must be rejected."""
        metadata_urls = [
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://test.metadata.google.internal/api",
            "http://metadata.azure.internal/metadata/instance",
            "http://instance-data/latest/meta-data",
        ]
        for url in metadata_urls:
            safe, err = is_safe_webhook_url(url)
            assert safe is False
            assert "cloud metadata" in err.lower() or "forbidden" in err.lower()

    def test_is_safe_webhook_url_blocks_rfc1918_private_ips(self) -> None:
        """Private subnets (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) must be blocked."""
        private_urls = [
            "http://10.0.0.1/report",
            "http://10.254.1.5:8080/webhook",
            "http://172.16.0.1/report",
            "http://172.31.255.255/report",
            "http://192.168.1.1/webhook",
            "http://192.168.100.50:9000/report",
        ]
        for url in private_urls:
            safe, err = is_safe_webhook_url(url, allow_local=False)
            assert safe is False, f"Failed to block private IP: {url}"
            assert "private" in err.lower() or "blocked" in err.lower()

    def test_is_safe_webhook_url_blocks_loopback_and_link_local(self) -> None:
        """Loopback, IPv6 loopback, and link-local addresses must be blocked."""
        blocked_urls = [
            "http://127.0.0.1:8000/report",
            "http://127.10.20.30/webhook",
            "http://localhost:8080/report",
            "http://api.localhost/report",
            "http://service.local/webhook",
            "http://[::1]/report",
            "http://[::ffff:127.0.0.1]/report",
            "http://169.254.1.1/report",
            "http://[fe80::1]/webhook",
            "http://100.64.0.1/webhook",  # Carrier-grade NAT
        ]
        for url in blocked_urls:
            safe, err = is_safe_webhook_url(url, allow_local=False)
            assert safe is False, f"Failed to block loopback/local URL: {url}"

    def test_is_safe_webhook_url_allows_local_when_flagged(self) -> None:
        """Localhost and private IPs must be allowed when allow_local=True."""
        local_urls = [
            "http://127.0.0.1:8000/report",
            "http://localhost:8000/report",
            "http://192.168.1.50:3000/webhook",
            "http://10.0.0.5:8080/report",
        ]
        for url in local_urls:
            # For localhost domain name, mock getaddrinfo returning 127.0.0.1
            mock_addr = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 8000))]
            with patch("socket.getaddrinfo", return_value=mock_addr):
                safe, err = is_safe_webhook_url(url, allow_local=True)
                assert safe is True, f"Failed to allow local URL with allow_local=True: {url} ({err})"
                assert err is None

    def test_is_safe_webhook_url_invalid_schemes(self) -> None:
        """Non-HTTP/HTTPS schemes and invalid inputs must be blocked."""
        invalid_inputs = [
            "file:///etc/passwd",
            "gopher://127.0.0.1:70",
            "ftp://example.com/file",
            "ws://example.com/socket",
            "",
            "   ",
            None,
            12345,
            "http://",
        ]
        for item in invalid_inputs:
            safe, err = is_safe_webhook_url(item)  # type: ignore
            assert safe is False
            assert err is not None

    def test_is_safe_webhook_url_domain_whitelist(self) -> None:
        """Domain whitelist (allowed_domains) must strictly restrict targets."""
        allowed = ["example.com", "trusted-partner.org"]
        mock_addr = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

        with patch("socket.getaddrinfo", return_value=mock_addr):
            # Exact match
            safe, err = is_safe_webhook_url("https://example.com/webhook", allowed_domains=allowed)
            assert safe is True

            # Subdomain match
            safe, err = is_safe_webhook_url("https://api.sub.example.com/reports", allowed_domains=allowed)
            assert safe is True

            # Non-whitelisted domain
            safe, err = is_safe_webhook_url("https://untrusted.com/webhook", allowed_domains=allowed)
            assert safe is False
            assert "not in allowed_domains" in err

    def test_is_safe_webhook_url_domain_whitelist_normalization(self) -> None:
        """Allowed domains with leading dots or uppercase should be normalized."""
        allowed = [".EXAMPLE.COM ", " TRUSTED.ORG "]
        mock_addr = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

        with patch("socket.getaddrinfo", return_value=mock_addr):
            safe, err = is_safe_webhook_url("https://api.example.com/webhook", allowed_domains=allowed)
            assert safe is True
            assert err is None

            safe, err = is_safe_webhook_url("https://trusted.org/webhook", allowed_domains=allowed)
            assert safe is True

    def test_is_safe_webhook_url_with_custom_port(self) -> None:
        """Verify handling of valid custom ports."""
        mock_addr = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 8443))]
        with patch("socket.getaddrinfo", return_value=mock_addr) as mock_gai:
            safe, err = is_safe_webhook_url("https://example.com:8443/webhook")
            assert safe is True
            # Verify port passed to getaddrinfo is 8443
            assert mock_gai.call_args[0][1] == 8443

    def test_is_safe_webhook_url_dns_rebinding_attack(self) -> None:
        """DNS rebinding attack where domain resolves to private/loopback/metadata IP must be blocked."""
        # 1. Domain resolving to 127.0.0.1
        mock_addr_loopback = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        with patch("socket.getaddrinfo", return_value=mock_addr_loopback):
            safe, err = is_safe_webhook_url("https://rebind-attack.com/webhook", allow_local=False)
            assert safe is False
            assert "dns rebinding" in err.lower()
            assert "127.0.0.1" in err

        # 2. Domain resolving to RFC 1918 private IP
        mock_addr_private = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.25", 443))]
        with patch("socket.getaddrinfo", return_value=mock_addr_private):
            safe, err = is_safe_webhook_url("https://rebind-internal.com/webhook", allow_local=False)
            assert safe is False
            assert "dns rebinding" in err.lower()
            assert "10.0.0.25" in err

        # 3. Domain resolving to cloud metadata IP
        mock_addr_metadata = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))]
        with patch("socket.getaddrinfo", return_value=mock_addr_metadata):
            safe, err = is_safe_webhook_url("https://rebind-metadata.com/webhook", allow_local=True)
            assert safe is False
            assert "cloud metadata" in err.lower() or "dns rebinding" in err.lower()

    def test_is_safe_webhook_url_dns_resolution_failure(self) -> None:
        """DNS resolution failures must result in safe=False without raising unhandled exceptions."""
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
            safe, err = is_safe_webhook_url("https://nonexistent-host-xyz-abc.com/webhook")
            assert safe is False
            assert "dns resolution failed" in err.lower()

    def test_is_safe_webhook_url_invalid_and_out_of_range_ports(self) -> None:
        """Invalid or out-of-range port numbers must be rejected cleanly without unhandled exceptions."""
        bad_port_urls = [
            "http://example.com:99999/webhook",
            "http://example.com:abc/webhook",
            "http://example.com:0/webhook",
            "http://example.com:70000/webhook",
        ]
        for url in bad_port_urls:
            safe, err = is_safe_webhook_url(url)
            assert safe is False
            assert "port" in err.lower() or "invalid url" in err.lower()

    def test_is_safe_webhook_url_blocks_aws_ipv6_metadata(self) -> None:
        """AWS IPv6 IMDS (fd00:ec2::254) must be blocked unconditionally even with allow_local=True."""
        safe, err = is_safe_webhook_url("http://[fd00:ec2::254]/latest/meta-data/", allow_local=True)
        assert safe is False
        assert "cloud metadata" in err.lower() or "forbidden" in err.lower()

    def test_is_safe_webhook_url_blocks_alternative_ip_encodings(self) -> None:
        """Decimal integer IP encodings resolving to loopback must be blocked."""
        # 2130706433 is 127.0.0.1 in decimal integer format
        safe, err = is_safe_webhook_url("http://2130706433/report", allow_local=False)
        assert safe is False
        assert "loopback" in err.lower() or "blocked" in err.lower() or "127.0.0.1" in err

    def test_is_safe_webhook_url_blocks_multicast_and_unspecified_even_if_allow_local(self) -> None:
        """Multicast and unspecified IPs must be rejected even when allow_local=True."""
        for url in ["http://0.0.0.0:8000/report", "http://224.0.0.1/report"]:
            safe, err = is_safe_webhook_url(url, allow_local=True)
            assert safe is False
            assert "blocked" in err.lower()


# ==============================================================================
# 2. Payload Builder Tests
# ==============================================================================
class TestPayloadBuilder:
    """Tests for application-agnostic session telemetry payload assembly."""

    def test_build_payload_standard_identifiers(self, minimal_session_config: SessionConfig) -> None:
        """Verify standard identifiers, agent identity, and timestamps are populated."""
        mock_ctx = MagicMock()
        mock_ctx.job.id = "job_lk_001"
        mock_ctx.room.name = "room_lk_alpha"

        payload = build_session_telemetry_payload(mock_ctx, minimal_session_config)

        assert payload["job_id"] == "job_lk_001"
        assert payload["room_name"] == "room_lk_alpha"
        # session_id resolved from metadata['session_id']
        assert payload["session_id"] == "test-sess-1234"
        assert payload["agent"] == {"name": "support-agent", "role": "customer_support"}
        assert payload["termination_reason"] == "normal"
        assert payload["ended_at"].endswith("Z")
        assert "error" not in payload

    def test_build_payload_session_id_precedence(self) -> None:
        """Test fallback precedence: metadata.session_id -> ctx.job.id -> agent.name."""
        cfg_no_sess_id = SessionConfig(
            agent=AgentConfig(name="fallback-agent", role="assistant"),
            models=ModelsConfig(provider="gemini", model="gemini-3.8-live"),
            instructions=InstructionsConfig(system_prompt="Prompt"),
            voice=VoiceConfig(provider="gemini", voice_id="Aoede"),
            metadata={},
        )

        mock_ctx = MagicMock()
        mock_ctx.job.id = "job_fallback_88"
        mock_ctx.room.name = "room_88"

        # Fallback to ctx.job.id
        p1 = build_session_telemetry_payload(mock_ctx, cfg_no_sess_id)
        assert p1["session_id"] == "job_fallback_88"

        # Fallback to agent.name when ctx has no job
        p2 = build_session_telemetry_payload(None, cfg_no_sess_id)
        assert p2["session_id"] == "fallback-agent"

    def test_build_payload_chat_history_extraction(self, minimal_session_config: SessionConfig) -> None:
        """Verify chat history extraction into normalized turns."""
        mock_report = MagicMock()
        mock_report.chat_history = [
            {"role": "user", "content": "Hello, I need help.", "created_at": 1700000001.0},
            {"role": "assistant", "content": ["I can assist you with that."], "created_at": 1700000002.5},
        ]
        mock_report.model_usage = None
        mock_report.events = None
        mock_report.audio_recording_path = None
        mock_report.started_at = 1700000000.0
        mock_report.duration = 15.2

        payload = build_session_telemetry_payload(None, minimal_session_config, lk_report=mock_report)

        assert payload["started_at"] is not None
        assert payload["duration_seconds"] == 15.2
        assert len(payload["chat_history"]) == 2
        assert payload["chat_history"][0] == {
            "role": "user",
            "content": "Hello, I need help.",
            "timestamp": 1700000001.0,
        }
        assert payload["chat_history"][1] == {
            "role": "assistant",
            "content": "I can assist you with that.",
            "timestamp": 1700000002.5,
        }

    def test_build_payload_real_livekit_chat_context(self, minimal_session_config: SessionConfig) -> None:
        """Verify extraction from a genuine LiveKit ChatContext object."""
        from livekit.agents.llm.chat_context import ChatContext

        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content="Hello from LiveKit participant")
        chat_ctx.add_message(role="assistant", content="Hello! How can I assist you?")

        mock_report = MagicMock()
        mock_report.chat_history = chat_ctx
        mock_report.model_usage = None
        mock_report.events = None
        mock_report.audio_recording_path = None
        mock_report.started_at = None
        mock_report.duration = None

        payload = build_session_telemetry_payload(None, minimal_session_config, lk_report=mock_report)
        assert len(payload["chat_history"]) == 2
        assert payload["chat_history"][0]["role"] == "user"
        assert payload["chat_history"][0]["content"] == "Hello from LiveKit participant"
        assert payload["chat_history"][0]["timestamp"] is not None
        assert payload["chat_history"][1]["role"] == "assistant"
        assert payload["chat_history"][1]["content"] == "Hello! How can I assist you?"

    def test_build_payload_chat_history_omitted_when_disabled(self, minimal_session_config: SessionConfig) -> None:
        """When include_chat_history=False, chat_history must be None."""
        minimal_session_config.reporting.payload_config.include_chat_history = False
        mock_report = MagicMock()
        mock_report.chat_history = [{"role": "user", "content": "Ignored"}]

        payload = build_session_telemetry_payload(None, minimal_session_config, lk_report=mock_report)
        assert payload["chat_history"] is None

    def test_build_payload_model_usage_metrics(self, minimal_session_config: SessionConfig) -> None:
        """Verify model usage extraction with token counts and cost metrics."""
        mock_usage_item = MagicMock()
        mock_usage_item.model_dump.return_value = {
            "model": "gemini-3.8-live",
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
            "cost_usd": 0.0015,
        }
        mock_report = MagicMock()
        mock_report.model_usage = [mock_usage_item]
        mock_report.chat_history = None
        mock_report.events = None

        payload = build_session_telemetry_payload(None, minimal_session_config, lk_report=mock_report)
        assert payload["model_usage"] == [{
            "model": "gemini-3.8-live",
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
            "cost_usd": 0.0015,
        }]

    def test_build_payload_events_and_audio_recording(self, minimal_session_config: SessionConfig) -> None:
        """Verify events timeline and audio recording path inclusion."""
        mock_event = MagicMock()
        mock_event.model_dump.return_value = {
            "type": "user_speech_committed",
            "timestamp": 1700000005.0,
        }
        mock_report = MagicMock()
        mock_report.chat_history = None
        mock_report.model_usage = None
        mock_report.events = [mock_event]
        mock_report.audio_recording_path = Path("/var/recordings/call_123.ogg")

        payload = build_session_telemetry_payload(None, minimal_session_config, lk_report=mock_report)
        assert payload["events"] == [{"type": "user_speech_committed", "timestamp": 1700000005.0}]
        assert payload["audio_recording_path"] is not None
        assert "call_123.ogg" in payload["audio_recording_path"]

    def test_build_payload_custom_fields_and_metadata_passthrough(
        self, minimal_session_config: SessionConfig
    ) -> None:
        """Verify custom_fields and metadata pass through unmodified."""
        payload = build_session_telemetry_payload(None, minimal_session_config)
        assert payload["custom_fields"] == {"tenant": "acme_corp", "tier": "gold"}
        assert payload["metadata"] == {"session_id": "test-sess-1234", "account_id": "acc-99"}

    def test_build_payload_error_termination(self, minimal_session_config: SessionConfig) -> None:
        """When an exception is passed, termination_reason is 'error' and error diagnostics are included."""
        err = ConnectionResetError("Remote server closed connection unexpectedly")
        payload = build_session_telemetry_payload(None, minimal_session_config, error=err)

        assert payload["termination_reason"] == "error"
        assert payload["error"] == {
            "type": "ConnectionResetError",
            "message": "Remote server closed connection unexpectedly",
        }

    def test_build_payload_zero_domain_assumptions(self, minimal_session_config: SessionConfig) -> None:
        """Assert payload schema contains zero domain-specific words."""
        payload = build_session_telemetry_payload(None, minimal_session_config)
        forbidden_terms = ["trainee", "dental", "rubric", "citrus", "marcus", "sop", "doctor"]
        serialized = json.dumps(payload).lower()
        for term in forbidden_terms:
            assert term not in serialized, f"Forbidden domain term '{term}' found in payload!"

    def test_build_payload_with_dict_report(self, minimal_session_config: SessionConfig) -> None:
        """Verify build_session_telemetry_payload extracts all fields when lk_report is a dictionary."""
        dict_report = {
            "job_id": "job-dict-999",
            "room": "room-dict-main",
            "started_at": 1700000000.0,
            "duration": 45.5,
            "chat_history": [
                {"role": "user", "content": "Need assistance", "timestamp": 1700000001.0},
                {"role": "assistant", "content": "I am here to help.", "timestamp": 1700000002.0},
            ],
            "model_usage": [{"model": "gemini-3.8-live", "total_tokens": 150}],
            "events": [{"type": "user_started_speaking"}],
            "audio_recording_path": "/recordings/dict_call.ogg",
        }

        payload = build_session_telemetry_payload(None, minimal_session_config, lk_report=dict_report)

        assert payload["job_id"] == "job-dict-999"
        assert payload["room_name"] == "room-dict-main"
        assert payload["duration_seconds"] == 45.5
        assert len(payload["chat_history"]) == 2
        assert payload["chat_history"][0]["content"] == "Need assistance"
        assert payload["model_usage"] == [{"model": "gemini-3.8-live", "total_tokens": 150}]
        assert payload["events"] == [{"type": "user_started_speaking"}]
        assert payload["audio_recording_path"] == "/recordings/dict_call.ogg"

    def test_build_payload_livekit_function_calling_turns(
        self, minimal_session_config: SessionConfig
    ) -> None:
        """Verify chat history correctly parses LiveKit FunctionCall and FunctionCallOutput items."""
        from livekit.agents.llm.chat_context import ChatContext, FunctionCall, FunctionCallOutput

        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content="Check my order 12345")
        fc = FunctionCall(
            id="call_01",
            call_id="c_order_1",
            name="lookup_order",
            arguments='{"order_id": "12345"}',
        )
        chat_ctx.items.append(fc)
        fco = FunctionCallOutput(
            id="out_01",
            call_id="c_order_1",
            output="Order shipped",
            is_error=False,
        )
        chat_ctx.items.append(fco)
        chat_ctx.add_message(role="assistant", content="Your order 12345 has shipped.")

        mock_report = MagicMock()
        mock_report.chat_history = chat_ctx
        mock_report.model_usage = None
        mock_report.events = None

        payload = build_session_telemetry_payload(None, minimal_session_config, lk_report=mock_report)

        turns = payload["chat_history"]
        assert len(turns) == 4
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Check my order 12345"
        assert turns[1]["role"] == "assistant"
        assert "Tool call: lookup_order" in turns[1]["content"]
        assert turns[2]["role"] == "tool"
        assert turns[2]["content"] == "Order shipped"
        assert turns[3]["role"] == "assistant"
        assert turns[3]["content"] == "Your order 12345 has shipped."

    def test_extract_chat_history_from_dict_and_string_list(self) -> None:
        """Verify _extract_chat_history handles dict turns and string lists."""
        from agent_framework.reporting.payload_builder import _extract_chat_history

        # Dict with items list
        d1 = {"items": [{"role": "user", "content": "Hello"}]}
        t1 = _extract_chat_history(d1)
        assert t1 == [{"role": "user", "content": "Hello", "timestamp": None}]

        # Single dict message
        d2 = {"role": "assistant", "content": "How can I help?"}
        t2 = _extract_chat_history(d2)
        assert t2 == [{"role": "assistant", "content": "How can I help?", "timestamp": None}]

        # List of plain strings
        s_list = ["User turn", "Assistant reply"]
        t3 = _extract_chat_history(s_list)
        assert t3 == [
            {"role": "user", "content": "User turn", "timestamp": None},
            {"role": "user", "content": "Assistant reply", "timestamp": None},
        ]


# ==============================================================================
# 3. Authentication Header Generation Tests
# ==============================================================================
class TestAuthHeaders:
    """Tests for Bearer, API Key, and HMAC-SHA256 signature generation."""

    def test_generate_auth_headers_bearer_token(self) -> None:
        """Bearer auth sets 'Authorization: Bearer <token>'."""
        auth = ReportingAuthConfig(type="bearer", secret_ref="env:WEBHOOK_TOKEN")
        env = {"WEBHOOK_TOKEN": "secret-bearer-jwt-token"}
        headers = generate_auth_headers(auth, b"{}", env=env)
        assert headers == {"Authorization": "Bearer secret-bearer-jwt-token"}

    def test_generate_auth_headers_bearer_token_already_prefixed(self) -> None:
        """Bearer auth should not duplicate 'Bearer ' prefix if already present."""
        auth = ReportingAuthConfig(type="bearer", secret_ref="env:WEBHOOK_TOKEN")
        env = {"WEBHOOK_TOKEN": "Bearer pre-prefixed-jwt"}
        headers = generate_auth_headers(auth, b"{}", env=env)
        assert headers == {"Authorization": "Bearer pre-prefixed-jwt"}

    def test_generate_auth_headers_api_key(self) -> None:
        """API key auth sets 'X-API-Key: <key>'."""
        auth = ReportingAuthConfig(type="api_key", secret_ref="env:MY_API_KEY")
        env = {"MY_API_KEY": "live_key_987654321"}
        headers = generate_auth_headers(auth, b"{}", env=env)
        assert headers == {"X-API-Key": "live_key_987654321"}

    def test_generate_auth_headers_hmac_sha256(self) -> None:
        """HMAC-SHA256 auth signs the payload body with the secret key."""
        auth = ReportingAuthConfig(type="hmac_sha256", secret_ref="env:WEBHOOK_SECRET")
        secret = "super-secret-hmac-key"
        payload_bytes = b'{"job_id": "test-123", "status": "completed"}'
        env = {"WEBHOOK_SECRET": secret}

        headers = generate_auth_headers(auth, payload_bytes, env=env)

        expected_sig = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        assert headers["X-Signature-SHA256"] == f"sha256={expected_sig}"
        assert headers["X-Signature"] == expected_sig

    def test_generate_auth_headers_missing_env_secret_raises(self) -> None:
        """Referencing a missing environment variable must raise SecretResolutionError."""
        auth = ReportingAuthConfig(type="bearer", secret_ref="env:NONEXISTENT_VAR")
        with pytest.raises(SecretResolutionError):
            generate_auth_headers(auth, b"{}", env={})

    def test_generate_auth_headers_none_or_empty(self) -> None:
        """When auth is None, returns empty header dictionary."""
        assert generate_auth_headers(None, b"{}") == {}


# ==============================================================================
# 4. Dispatcher Retries & Delivery Tests
# ==============================================================================
class TestReportingDispatcher:
    """Tests for asynchronous HTTP dispatch, retries, and dead-letter logging."""

    @pytest.mark.asyncio
    async def test_dispatcher_successful_post(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """Successful POST returns DeliveryResult with success=True and attempts=1."""
        mock_session = MockHttpSession([MockHttpResponse(200, '{"ok": true}')])
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
        )

        # Mock SSRF validation as safe
        with patch("agent_framework.reporting.dispatcher.is_safe_webhook_url", return_value=(True, None)):
            result = await dispatcher.dispatch({"session_id": "sess-01", "room": "r1"})

        assert result.success is True
        assert result.status_code == 200
        assert result.attempts == 1
        assert result.dead_letter_path is None
        assert mock_session.call_count == 1
        assert mock_session.calls[0]["method"] == "POST"

    @pytest.mark.asyncio
    async def test_dispatcher_supports_put_method(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """When endpoint method is PUT, dispatcher issues an HTTP PUT."""
        minimal_session_config.reporting.endpoint.method = "PUT"  # type: ignore
        mock_session = MockHttpSession([MockHttpResponse(200, "Updated")])
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
        )

        with patch("agent_framework.reporting.dispatcher.is_safe_webhook_url", return_value=(True, None)):
            result = await dispatcher.dispatch({"session_id": "sess-put"})

        assert result.success is True
        assert mock_session.calls[0]["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_dispatcher_retries_on_503_and_succeeds(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """Dispatcher retries on 503 errors with exponential backoff and succeeds on attempt 3."""
        # 503 -> 503 -> 200
        responses = [
            MockHttpResponse(503, "Service Unavailable"),
            MockHttpResponse(503, "Service Unavailable"),
            MockHttpResponse(200, "Delivered"),
        ]
        mock_session = MockHttpSession(responses)
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
        )

        with patch("agent_framework.reporting.dispatcher.is_safe_webhook_url", return_value=(True, None)):
            result = await dispatcher.dispatch({"session_id": "sess-retry"})

        assert result.success is True
        assert result.status_code == 200
        assert result.attempts == 3
        assert mock_session.call_count == 3
        assert result.dead_letter_path is None

    @pytest.mark.asyncio
    async def test_dispatcher_retries_on_network_timeout(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """Dispatcher retries when encountering asyncio.TimeoutError."""
        responses = [
            asyncio.TimeoutError("Connection timed out"),
            MockHttpResponse(200, "Success after timeout"),
        ]
        mock_session = MockHttpSession(responses)
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
        )

        with patch("agent_framework.reporting.dispatcher.is_safe_webhook_url", return_value=(True, None)):
            result = await dispatcher.dispatch({"session_id": "sess-timeout"})

        assert result.success is True
        assert result.attempts == 2
        assert mock_session.call_count == 2

    @pytest.mark.asyncio
    async def test_dispatcher_fail_fast_on_client_error_400(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """4xx client errors (e.g. 400 Bad Request) must fail immediately without retrying."""
        mock_session = MockHttpSession([MockHttpResponse(400, "Bad Request: Invalid JSON Schema")])
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
        )

        with patch("agent_framework.reporting.dispatcher.is_safe_webhook_url", return_value=(True, None)):
            result = await dispatcher.dispatch({"session_id": "sess-400"})

        assert result.success is False
        assert result.status_code == 400
        assert result.attempts == 1  # Did not retry!
        assert mock_session.call_count == 1
        assert result.dead_letter_path is not None
        assert Path(result.dead_letter_path).exists()

    @pytest.mark.asyncio
    async def test_dispatcher_dead_letter_on_exhausted_retries(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """When all retries fail, writes dead-letter file without raising unhandled exceptions."""
        # max_retries = 2 -> total attempts = 3
        responses = [
            MockHttpResponse(500, "Internal Server Error"),
            MockHttpResponse(502, "Bad Gateway"),
            MockHttpResponse(503, "Service Unavailable"),
        ]
        mock_session = MockHttpSession(responses)
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
        )

        with patch("agent_framework.reporting.dispatcher.is_safe_webhook_url", return_value=(True, None)):
            result = await dispatcher.dispatch({"session_id": "sess-exhausted", "data": "important_metrics"})

        assert result.success is False
        assert result.status_code == 503
        assert result.attempts == 3
        assert result.dead_letter_path is not None

        # Verify dead letter contents
        dl_path = Path(result.dead_letter_path)
        assert dl_path.exists()
        with open(dl_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["attempts"] == 3
        assert data["target_url"] == "https://api.example.com/reports"
        assert data["payload"]["session_id"] == "sess-exhausted"
        assert data["error"]["status_code"] == 503
        assert "503" in data["error"]["reason"]

    @pytest.mark.asyncio
    async def test_dispatcher_ssrf_aborts_dispatch_and_writes_dead_letter(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """If target URL is rejected by SSRF validation, HTTP dispatch is never called."""
        minimal_session_config.reporting.endpoint.url = "http://169.254.169.254/latest/meta-data"  # type: ignore
        mock_session = MockHttpSession([MockHttpResponse(200, "Should never be called")])

        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
        )

        result = await dispatcher.dispatch({"session_id": "sess-ssrf"})

        assert result.success is False
        assert mock_session.call_count == 0  # CRITICAL: Network was NEVER accessed
        assert "ssrf" in result.error.lower()
        assert result.dead_letter_path is not None
        assert Path(result.dead_letter_path).exists()

    @pytest.mark.asyncio
    async def test_dispatcher_no_unhandled_exceptions_on_malformed_url(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """Malformed URL returns DeliveryResult(success=False) without unhandled crash."""
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            dead_letter_dir=tmp_path,
        )
        result = await dispatcher.dispatch({"session_id": "sess-bad"}, url_override="not_a_valid_url")
        assert result.success is False
        assert result.dead_letter_path is not None

    @pytest.mark.asyncio
    async def test_send_session_report_convenience_helper(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """End-to-end test of send_session_report convenience function."""
        mock_ctx = MagicMock()
        mock_ctx.job.id = "job-e2e"
        mock_ctx.room.name = "room-e2e"

        mock_session = MockHttpSession([MockHttpResponse(200, '{"received": true}')])

        with patch("agent_framework.reporting.dispatcher.is_safe_webhook_url", return_value=(True, None)):
            result = await send_session_report(
                ctx=mock_ctx,
                session_config=minimal_session_config,
                session=mock_session,  # type: ignore
                dead_letter_dir=tmp_path,
            )

        assert result.success is True
        assert result.status_code == 200
        assert mock_session.call_count == 1
        call_kwargs = mock_session.calls[0]["kwargs"]
        sent_payload = json.loads(call_kwargs["data"].decode("utf-8"))
        assert sent_payload["job_id"] == "job-e2e"
        assert sent_payload["room_name"] == "room-e2e"
        assert sent_payload["agent"]["name"] == "support-agent"

    @pytest.mark.asyncio
    async def test_dispatcher_safe_redirect_followed(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """When receiving a 302 to a safe target, dispatcher follows the redirect safely."""
        responses = [
            MockHttpResponse(302, "Found", headers={"Location": "https://api.example.com/v2/reports"}),
            MockHttpResponse(200, '{"redirect_success": true}'),
        ]
        mock_session = MockHttpSession(responses)
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
        )

        with patch("agent_framework.reporting.dispatcher.is_safe_webhook_url", return_value=(True, None)):
            result = await dispatcher.dispatch({"session_id": "sess-redirect"})

        assert result.success is True
        assert result.status_code == 200
        assert mock_session.call_count == 2
        assert mock_session.calls[1]["url"] == "https://api.example.com/v2/reports"

    @pytest.mark.asyncio
    async def test_dispatcher_ssrf_blocks_unsafe_redirect(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """When receiving a 302 pointing to cloud metadata, SSRF validation aborts immediately."""
        responses = [
            MockHttpResponse(
                302, "Found", headers={"Location": "http://169.254.169.254/latest/meta-data/"}
            ),
            MockHttpResponse(200, "Should NEVER be called"),
        ]
        mock_session = MockHttpSession(responses)
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
        )

        def mock_ssrf_side_effect(url: str, *args: Any, **kwargs: Any) -> tuple[bool, str | None]:
            if "169.254" in url:
                return False, "Cloud metadata IP address '169.254.169.254' is forbidden"
            return True, None

        with patch("agent_framework.reporting.dispatcher.is_safe_webhook_url", side_effect=mock_ssrf_side_effect):
            result = await dispatcher.dispatch({"session_id": "sess-redirect-attack"})

        assert result.success is False
        assert mock_session.call_count == 1  # Did NOT follow redirect to metadata!
        assert "ssrf" in result.error.lower()
        assert result.dead_letter_path is not None
        assert Path(result.dead_letter_path).exists()

    @pytest.mark.asyncio
    async def test_dispatcher_custom_headers_secret_substitution(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """Custom headers with {{env:KEY}} should be resolved from env mapping."""
        minimal_session_config.reporting.endpoint.headers = {  # type: ignore
            "X-Custom-Tenant": "acme",
            "X-Tenant-Key": "{{env:TENANT_SECRET}}",
        }
        mock_session = MockHttpSession([MockHttpResponse(200, "OK")])
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
            env={"TENANT_SECRET": "top-secret-val-42"},
        )

        with patch("agent_framework.reporting.dispatcher.is_safe_webhook_url", return_value=(True, None)):
            result = await dispatcher.dispatch({"session_id": "sess-headers"})

        assert result.success is True
        sent_headers = mock_session.calls[0]["kwargs"]["headers"]
        assert sent_headers["X-Custom-Tenant"] == "acme"
        assert sent_headers["X-Tenant-Key"] == "top-secret-val-42"

    @pytest.mark.asyncio
    async def test_dispatcher_custom_headers_missing_secret_dead_letters(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """When custom header references a missing secret, dispatch fails fast and dead-letters."""
        minimal_session_config.reporting.endpoint.headers = {  # type: ignore
            "X-Tenant-Key": "{{env:MISSING_SECRET_KEY}}",
        }
        mock_session = MockHttpSession([MockHttpResponse(200, "OK")])
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            session=mock_session,  # type: ignore
            dead_letter_dir=tmp_path,
            env={},
        )

        result = await dispatcher.dispatch({"session_id": "sess-missing-secret"})

        assert result.success is False
        assert mock_session.call_count == 0  # Did not send request
        assert "MISSING_SECRET_KEY" in result.error
        assert result.dead_letter_path is not None
        assert Path(result.dead_letter_path).exists()

    @pytest.mark.asyncio
    async def test_dispatcher_dead_letter_truncates_long_session_id(
        self, minimal_session_config: SessionConfig, tmp_path: Path
    ) -> None:
        """Extremely long session IDs must be truncated so filenames do not exceed filesystem limits."""
        dispatcher = ReportingDispatcher(
            config=minimal_session_config.reporting,
            dead_letter_dir=tmp_path,
        )
        long_id = "A" * 300
        result = await dispatcher.dispatch({"session_id": long_id}, url_override="http://invalid_domain_that_fails_ssrf")

        assert result.success is False
        assert result.dead_letter_path is not None
        filename = Path(result.dead_letter_path).name
        assert len(filename) < 150
        assert Path(result.dead_letter_path).exists()
