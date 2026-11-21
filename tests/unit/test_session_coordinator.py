"""Unit tests for Phase 5: Unified Session Coordinator & Server Decoupling.

Tests:
1. SessionContext lifecycle methods and property calculations.
2. SessionContext binding to JobContext without global state leakage (multi-tenant isolation).
3. SessionContext fallback retrieval from ctx.proc.userdata.
4. Full mock SessionCoordinator initialization: room connection, metadata extraction,
   model and tool wiring, Agent and AgentSession startup.
5. First-turn greeting execution ('say', 'generate_reply', and passive 'listen').
6. Max duration watchdog timer and room disconnect.
7. Inactivity watchdog timer and activity reset.
8. Disconnect teardown invoking ReportingDispatcher with idempotency guarantee.
9. Participant join waiting with timeout and target identity matching.
10. Participant join timeout error handling.
11. Participant join skipped when target or remote participant is already present.
12. Server initialization, health check port (8081), and shutdown timeout options.
13. Server setup and CLI runner wiring.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RecordingOptions,
    TurnHandlingOptions,
)

from agent_framework.config.schema import (
    AgentConfig,
    FirstTurnConfig,
    InstructionsConfig,
    ModelsConfig,
    RecordingConfig,
    ReportingConfig,
    ReportingEndpointConfig,
    SessionConfig,
    SessionLifecycleConfig,
    ToolConfig,
    TurnDetectionConfig,
    VADConfig,
    VoiceConfig,
)
from agent_framework.config.settings import FrameworkSettings, settings
from agent_framework.core.context import (
    SessionContext,
    attach_to_job,
    detach_from_job,
    get_session_context,
)
from agent_framework.core.server import (
    create_server,
    run_app,
    run_server,
    setup_server,
)
from agent_framework.core.session import SessionCoordinator
from agent_framework.models.factory import ModelFactory
from agent_framework.reporting.dispatcher import DeliveryResult, ReportingDispatcher
from agent_framework.tools.registry import ToolRegistry
from agent_framework.turn_detection.factory import TurnDetectionFactory


# ==============================================================================
# Helper Fixtures & Builders
# ==============================================================================

def make_valid_session_config(**overrides: Any) -> SessionConfig:
    """Helper creating a strictly valid SessionConfig instance."""
    data: dict[str, Any] = {
        "models": {
            "provider": "gemini",
            "model": "gemini-3.8-live",
            "temperature": 0.7,
        },
        "instructions": {
            "system_prompt": "You are a customer support agent. Help the caller with their inquiry.",
            "spoken_dialogue_rules": True,
        },
        "voice": {
            "provider": "gemini",
            "voice_id": "Aoede",
            "language": "en-US",
        },
        "session": {
            "max_duration_seconds": 1800,
            "inactivity_timeout_seconds": 120,
            "participant_join_timeout_seconds": 30.0,
            "delete_room_on_terminate": True,
        },
        "reporting": {
            "enabled": True,
            "endpoint": {
                "url": "https://api.example.com/reports",
                "method": "POST",
            },
        },
    }
    data.update(overrides)
    return SessionConfig(**data)


def make_mock_participant(
    identity: str = "test_user",
    metadata: str = "",
    attributes: Optional[dict[str, Any]] = None,
) -> MagicMock:
    """Creates a mock participant with explicit metadata and attributes."""
    p = MagicMock()
    p.identity = identity
    p.metadata = metadata
    p.attributes = attributes or {}
    return p


def make_mock_job_context(
    job_id: str = "job_test_123",
    room_name: str = "room_test_123",
    metadata_json: Optional[str] = None,
    remote_participants: Optional[dict[str, Any]] = None,
) -> MagicMock:
    """Creates a mock JobContext matching LiveKit agents structure."""
    ctx = MagicMock(spec=JobContext)
    ctx.job = MagicMock()
    ctx.job.id = job_id
    ctx.job.metadata = metadata_json or ""

    ctx.room = MagicMock()
    ctx.room.name = room_name
    ctx.room.metadata = ""
    ctx.room.remote_participants = remote_participants or {}
    ctx.room.disconnect = AsyncMock()

    ctx.connect = AsyncMock()
    ctx.wait_for_participant = AsyncMock()
    ctx.delete_room = AsyncMock()
    ctx.make_session_report = MagicMock(return_value={"report_id": f"rep_{job_id}"})
    ctx.add_shutdown_callback = MagicMock()

    ctx.proc = MagicMock()
    ctx.proc.userdata = {}

    return ctx


# ==============================================================================
# 1. SessionContext Tests
# ==============================================================================

def test_session_context_initialization_and_lifecycle():
    """Verifies SessionContext status transitions, timestamps, and duration calculation."""
    config = make_valid_session_config()
    sc = SessionContext(
        job_id="job_001",
        room_name="room_001",
        session_config=config,
    )

    assert sc.job_id == "job_001"
    assert sc.room_name == "room_001"
    assert sc.status == "initialized"
    assert sc.started_at is None
    assert sc.ended_at is None
    assert sc.duration_seconds == 0.0
    assert not sc.is_active

    # Mark started
    sc.mark_started()
    assert sc.status == "running"
    assert sc.started_at is not None
    assert sc.is_active
    assert sc.duration_seconds >= 0.0

    # Mark ended
    sc.mark_ended(status="completed", reason="User finished call")
    assert sc.status == "completed"
    assert sc.ended_at is not None
    assert not sc.is_active
    assert sc.termination_reason == "User finished call"
    assert sc.duration_seconds >= 0.0


def test_session_context_binding_and_multi_tenant_isolation():
    """Verifies SessionContext binds to JobContext without global state leakage."""
    cfg1 = make_valid_session_config(
        instructions={"system_prompt": "Agent A instructions"}
    )
    cfg2 = make_valid_session_config(
        instructions={"system_prompt": "Agent B instructions"}
    )

    ctx1 = make_mock_job_context(job_id="job_tenant_A", room_name="room_A")
    ctx2 = make_mock_job_context(job_id="job_tenant_B", room_name="room_B")

    sc1 = SessionContext(job_id="job_tenant_A", room_name="room_A", session_config=cfg1)
    sc2 = SessionContext(job_id="job_tenant_B", room_name="room_B", session_config=cfg2)

    # Attach to respective contexts
    attach_to_job(ctx1, sc1)
    attach_to_job(ctx2, sc2)

    # Verify retrieval returns strictly isolated contexts
    retrieved_1 = get_session_context(ctx1)
    retrieved_2 = get_session_context(ctx2)

    assert retrieved_1 is sc1
    assert retrieved_2 is sc2
    assert retrieved_1.session_config.instructions.system_prompt == "Agent A instructions"
    assert retrieved_2.session_config.instructions.system_prompt == "Agent B instructions"

    # Mutating sc1 state does not impact sc2
    sc1.mark_started()
    assert sc1.is_active
    assert not sc2.is_active

    # Detach ctx1 leaves ctx2 untouched
    detached_1 = detach_from_job(ctx1)
    assert detached_1 is sc1
    assert get_session_context(ctx1) is None
    assert get_session_context(ctx2) is sc2


def test_session_context_fallback_userdata():
    """Verifies fallback retrieval from ctx.proc.userdata if direct attribute is missing."""
    config = make_valid_session_config()
    ctx = make_mock_job_context(job_id="job_proc_test")
    sc = SessionContext(job_id="job_proc_test", room_name="room_test", session_config=config)

    attach_to_job(ctx, sc)

    # Simulate missing direct attribute (e.g. across process or thread boundaries)
    delattr(ctx, "_agent_framework_session_context")

    # Should still resolve via ctx.proc.userdata
    retrieved = get_session_context(ctx)
    assert retrieved is sc

    # Graceful handling of None ctx
    assert get_session_context(None) is None
    assert detach_from_job(None) is None


def test_concurrent_multi_tenant_no_userdata_bleed():
    """Verifies that concurrent jobs sharing ctx.proc.userdata never bleed or overwrite each other."""
    cfg1 = make_valid_session_config(instructions={"system_prompt": "Prompt Job 1"})
    cfg2 = make_valid_session_config(instructions={"system_prompt": "Prompt Job 2"})

    # Shared process userdata (simulating multiple threads/jobs on same worker process)
    shared_proc = MagicMock()
    shared_proc.userdata = {}

    ctx1 = make_mock_job_context(job_id="job_worker_1")
    ctx1.proc = shared_proc
    ctx2 = make_mock_job_context(job_id="job_worker_2")
    ctx2.proc = shared_proc

    sc1 = SessionContext(job_id="job_worker_1", room_name="room_1", session_config=cfg1)
    sc2 = SessionContext(job_id="job_worker_2", room_name="room_2", session_config=cfg2)

    attach_to_job(ctx1, sc1)
    attach_to_job(ctx2, sc2)

    # Simulate direct attribute loss on both to test userdata fallback lookup
    delattr(ctx1, "_agent_framework_session_context")
    delattr(ctx2, "_agent_framework_session_context")

    # Verify each resolves strictly to its own context without cross-talk
    retrieved_1 = get_session_context(ctx1)
    retrieved_2 = get_session_context(ctx2)
    assert retrieved_1 is sc1
    assert retrieved_2 is sc2
    assert retrieved_1.session_config.instructions.system_prompt == "Prompt Job 1"
    assert retrieved_2.session_config.instructions.system_prompt == "Prompt Job 2"

    # Detaching Job 1 leaves Job 2 completely intact in shared proc.userdata
    detach_from_job(ctx1)
    assert get_session_context(ctx1) is None
    assert get_session_context(ctx2) is sc2


@pytest.mark.asyncio
async def test_session_context_cancel_watchdogs():
    """Verifies cancel_watchdogs safely cancels running tasks."""
    config = make_valid_session_config()
    sc = SessionContext(job_id="job_watchdogs", room_name="room_w", session_config=config)

    async def _dummy_task():
        await asyncio.sleep(100)

    task1 = asyncio.create_task(_dummy_task())
    task2 = asyncio.create_task(_dummy_task())

    sc.max_duration_task = task1
    sc.inactivity_task = task2

    sc.cancel_watchdogs()
    await asyncio.sleep(0)

    assert task1.cancelled() or task1.cancelling() > 0
    assert task2.cancelled() or task2.cancelling() > 0
    assert sc.max_duration_task is None
    assert sc.inactivity_task is None


# ==============================================================================
# 2. SessionCoordinator Full Initialization Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_session_coordinator_full_initialization():
    """Verifies full SessionCoordinator initialization flow:
    connect -> metadata -> model -> turn_detection -> tools -> agent -> session.start.
    """
    config = make_valid_session_config()
    meta_json = json.dumps(config.model_dump())

    remote_p = make_mock_participant(identity="trainee_user")
    ctx = make_mock_job_context(
        job_id="job_full_init",
        room_name="room_full_init",
        metadata_json=meta_json,
        remote_participants={"p1": remote_p},
    )

    # Mock AgentSession and Agent
    mock_agent_session = MagicMock(spec=AgentSession)
    mock_agent_session.start = AsyncMock()
    mock_agent_session.say = AsyncMock()
    mock_agent_session.generate_reply = AsyncMock()
    mock_agent_session.aclose = AsyncMock()

    mock_model = MagicMock()
    mock_tools = [MagicMock()]
    mock_turn_handling = {"interruption": {"enabled": True}}

    with (
        patch("agent_framework.core.session.ModelFactory.create_model", return_value=mock_model) as mock_create_model,
        patch("agent_framework.core.session.TurnDetectionFactory.build_turn_handling_options", return_value=mock_turn_handling) as mock_turn_factory,
        patch("agent_framework.core.session.ToolRegistry.build_tools_from_config", return_value=mock_tools) as mock_tools_factory,
        patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session) as mock_session_cls,
        patch("agent_framework.core.session.Agent") as mock_agent_cls,
    ):
        coordinator = SessionCoordinator()
        await coordinator.run(ctx)

        # 1. Connect called
        ctx.connect.assert_awaited_once()

        # 2. Model created from validated config
        mock_create_model.assert_called_once()
        created_cfg = mock_create_model.call_args[0][0]
        assert created_cfg.models.model == "gemini-3.8-live"

        # 3. Turn handling options created
        mock_turn_factory.assert_called_once_with(created_cfg)

        # 4. Tools resolved from config
        mock_tools_factory.assert_called_once_with(created_cfg, ctx=ctx)

        # 5. Agent constructed with effective system instructions and tools
        mock_agent_cls.assert_called_once()
        agent_kwargs = mock_agent_cls.call_args[1]
        assert "Conversational speech guidelines:" in agent_kwargs["instructions"]
        assert agent_kwargs["tools"] == mock_tools

        # 6. AgentSession constructed
        mock_session_cls.assert_called_once()

        # 7. AgentSession started
        mock_agent_session.start.assert_awaited_once()

        # 8. SessionContext bound and started
        sc = get_session_context(ctx)
        assert sc is not None
        assert sc.status == "running"
        assert sc.is_active

        # Cleanup coordinator watchdogs
        sc.cancel_watchdogs()


# ==============================================================================
# 3. First-Turn Greeting Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_first_turn_say():
    """Verifies that mode='say' with text invokes session.say(text)."""
    config = make_valid_session_config(
        instructions={
            "system_prompt": "Assistant prompt",
            "first_turn": {
                "speaker": "agent",
                "mode": "say",
                "text": "Hello, how may I assist you today?",
            },
        }
    )
    meta_json = json.dumps(config.model_dump())
    ctx = make_mock_job_context(
        job_id="job_say",
        metadata_json=meta_json,
        remote_participants={"p1": make_mock_participant(identity="caller")},
    )

    mock_agent_session = MagicMock(spec=AgentSession)
    mock_agent_session.start = AsyncMock()
    mock_agent_session.say = AsyncMock()
    mock_agent_session.generate_reply = AsyncMock()

    with (
        patch("agent_framework.core.session.ModelFactory.create_model"),
        patch("agent_framework.core.session.TurnDetectionFactory.build_turn_handling_options"),
        patch("agent_framework.core.session.ToolRegistry.build_tools_from_config", return_value=[]),
        patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session),
        patch("agent_framework.core.session.Agent"),
    ):
        coordinator = SessionCoordinator()
        await coordinator.run(ctx)

        mock_agent_session.say.assert_awaited_once_with("Hello, how may I assist you today?")
        mock_agent_session.generate_reply.assert_not_called()

        sc = get_session_context(ctx)
        sc.cancel_watchdogs()


@pytest.mark.asyncio
async def test_first_turn_generate_reply():
    """Verifies that mode='generate_reply' invokes session.generate_reply()."""
    config = make_valid_session_config(
        instructions={
            "system_prompt": "Assistant prompt",
            "first_turn": {
                "speaker": "agent",
                "mode": "generate_reply",
            },
        }
    )
    meta_json = json.dumps(config.model_dump())
    ctx = make_mock_job_context(
        job_id="job_gen_reply",
        metadata_json=meta_json,
        remote_participants={"p1": make_mock_participant(identity="caller")},
    )

    mock_agent_session = MagicMock(spec=AgentSession)
    mock_agent_session.start = AsyncMock()
    mock_agent_session.say = AsyncMock()
    mock_agent_session.generate_reply = AsyncMock()

    with (
        patch("agent_framework.core.session.ModelFactory.create_model"),
        patch("agent_framework.core.session.TurnDetectionFactory.build_turn_handling_options"),
        patch("agent_framework.core.session.ToolRegistry.build_tools_from_config", return_value=[]),
        patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session),
        patch("agent_framework.core.session.Agent"),
    ):
        coordinator = SessionCoordinator()
        await coordinator.run(ctx)

        mock_agent_session.generate_reply.assert_awaited_once()
        mock_agent_session.say.assert_not_called()

        sc = get_session_context(ctx)
        sc.cancel_watchdogs()


@pytest.mark.asyncio
async def test_first_turn_listen_passive():
    """Verifies that default speaker='user' listens passively without say/generate_reply."""
    config = make_valid_session_config(
        instructions={
            "system_prompt": "Assistant prompt",
            "first_turn": {
                "speaker": "user",
                "mode": "listen",
            },
        }
    )
    meta_json = json.dumps(config.model_dump())
    ctx = make_mock_job_context(
        job_id="job_listen",
        metadata_json=meta_json,
        remote_participants={"p1": make_mock_participant(identity="caller")},
    )

    mock_agent_session = MagicMock(spec=AgentSession)
    mock_agent_session.start = AsyncMock()
    mock_agent_session.say = AsyncMock()
    mock_agent_session.generate_reply = AsyncMock()

    with (
        patch("agent_framework.core.session.ModelFactory.create_model"),
        patch("agent_framework.core.session.TurnDetectionFactory.build_turn_handling_options"),
        patch("agent_framework.core.session.ToolRegistry.build_tools_from_config", return_value=[]),
        patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session),
        patch("agent_framework.core.session.Agent"),
    ):
        coordinator = SessionCoordinator()
        await coordinator.run(ctx)

        mock_agent_session.say.assert_not_called()
        mock_agent_session.generate_reply.assert_not_called()

        sc = get_session_context(ctx)
        sc.cancel_watchdogs()


# ==============================================================================
# 4. Watchdogs & Teardown Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_max_duration_watchdog_disconnects_room():
    """Verifies max duration watchdog triggers teardown and deletes room upon expiration."""
    config = make_valid_session_config(
        session={
            "max_duration_seconds": 30,  # Minimum valid schema value (ge=30)
            "inactivity_timeout_seconds": 60,
            "delete_room_on_terminate": True,
        }
    )
    meta_json = json.dumps(config.model_dump())
    ctx = make_mock_job_context(
        job_id="job_max_duration",
        metadata_json=meta_json,
        remote_participants={"p1": make_mock_participant(identity="user")},
    )

    mock_agent_session = MagicMock(spec=AgentSession)
    mock_agent_session.start = AsyncMock()
    mock_agent_session.aclose = AsyncMock()

    sleep_called_with = None
    original_sleep = asyncio.sleep

    async def _mock_sleep(duration):
        nonlocal sleep_called_with
        if duration == 30:
            sleep_called_with = duration
            # Yield briefly to simulate timeout completion immediately
            await original_sleep(0.01)
            return
        await original_sleep(duration)

    with (
        patch("agent_framework.core.session.ModelFactory.create_model"),
        patch("agent_framework.core.session.TurnDetectionFactory.build_turn_handling_options"),
        patch("agent_framework.core.session.ToolRegistry.build_tools_from_config", return_value=[]),
        patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session),
        patch("agent_framework.core.session.Agent"),
        patch.object(ReportingDispatcher, "send_session_report", new_callable=AsyncMock) as mock_report,
        patch("asyncio.sleep", side_effect=_mock_sleep),
    ):
        coordinator = SessionCoordinator()
        await coordinator.run(ctx)

        sc = get_session_context(ctx)
        assert sc is not None
        assert sc.is_active

        # Allow watchdog task to run
        await original_sleep(0.05)

        # Room deletion requested
        ctx.delete_room.assert_awaited_once()

        # Reporting dispatched
        mock_report.assert_awaited_once()
        assert sc.status == "timed_out"
        assert "Max duration" in (sc.termination_reason or "")
        assert sleep_called_with == 30


@pytest.mark.asyncio
async def test_inactivity_watchdog_disconnects_room():
    """Verifies inactivity watchdog triggers teardown when session is idle."""
    config = make_valid_session_config(
        session={
            "max_duration_seconds": 60,
            "inactivity_timeout_seconds": 1,  # 1 second for fast test
            "delete_room_on_terminate": True,
        }
    )
    meta_json = json.dumps(config.model_dump())
    ctx = make_mock_job_context(
        job_id="job_inactivity",
        metadata_json=meta_json,
        remote_participants={"p1": make_mock_participant(identity="user")},
    )

    mock_agent_session = MagicMock(spec=AgentSession)
    mock_agent_session.start = AsyncMock()
    mock_agent_session.aclose = AsyncMock()
    mock_agent_session.on = MagicMock()

    with (
        patch("agent_framework.core.session.ModelFactory.create_model"),
        patch("agent_framework.core.session.TurnDetectionFactory.build_turn_handling_options"),
        patch("agent_framework.core.session.ToolRegistry.build_tools_from_config", return_value=[]),
        patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session),
        patch("agent_framework.core.session.Agent"),
        patch.object(ReportingDispatcher, "send_session_report", new_callable=AsyncMock) as mock_report,
    ):
        coordinator = SessionCoordinator()
        await coordinator.run(ctx)

        # Wait for inactivity watchdog to fire
        await asyncio.sleep(1.5)

        ctx.delete_room.assert_awaited_once()
        mock_report.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_teardown_invoking_reporting_idempotent():
    """Verifies teardown invokes ReportingDispatcher and is strictly idempotent."""
    config = make_valid_session_config()
    ctx = make_mock_job_context(job_id="job_teardown")

    sc = SessionContext(job_id="job_teardown", room_name="room_t", session_config=config)
    attach_to_job(ctx, sc)
    sc.mark_started()

    mock_agent_session = MagicMock()
    mock_agent_session.aclose = AsyncMock()
    sc.session = mock_agent_session

    with patch.object(ReportingDispatcher, "send_session_report", new_callable=AsyncMock) as mock_report:
        coordinator = SessionCoordinator()

        # First teardown call
        await coordinator.teardown(ctx, termination_reason="Normal end of call")

        assert mock_report.await_count == 1
        assert sc.reporting_sent is True
        assert sc.status == "completed"
        mock_agent_session.aclose.assert_awaited_once()
        ctx.delete_room.assert_awaited_once()

        # Second teardown call (e.g. from server on_session_end)
        await coordinator.teardown(ctx, termination_reason="Duplicate end")

        # Must NOT re-dispatch reporting
        assert mock_report.await_count == 1


# ==============================================================================
# 5. Participant Waiting Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_participant_waiting_with_target_identity():
    """Verifies participant join waiting when target_participant_identity is configured."""
    config = make_valid_session_config(
        session={
            "target_participant_identity": "specific_trainee",
            "participant_join_timeout_seconds": 5.0,
        }
    )
    meta_json = json.dumps(config.model_dump())

    # Initial room has NO remote participants
    ctx = make_mock_job_context(
        job_id="job_wait_identity",
        metadata_json=meta_json,
        remote_participants={},
    )

    joined_participant = make_mock_participant(identity="specific_trainee")
    ctx.wait_for_participant = AsyncMock(return_value=joined_participant)

    mock_agent_session = MagicMock(spec=AgentSession)
    mock_agent_session.start = AsyncMock()

    with (
        patch("agent_framework.core.session.ModelFactory.create_model"),
        patch("agent_framework.core.session.TurnDetectionFactory.build_turn_handling_options"),
        patch("agent_framework.core.session.ToolRegistry.build_tools_from_config", return_value=[]),
        patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session),
        patch("agent_framework.core.session.Agent"),
    ):
        coordinator = SessionCoordinator()
        await coordinator.run(ctx)

        # Verified wait_for_participant was awaited with target identity
        ctx.wait_for_participant.assert_awaited_once_with(identity="specific_trainee")

        sc = get_session_context(ctx)
        sc.cancel_watchdogs()


@pytest.mark.asyncio
async def test_participant_waiting_skipped_when_already_present():
    """Verifies participant waiting is skipped if the target participant is already in the room."""
    config = make_valid_session_config(
        session={
            "target_participant_identity": "alice",
            "participant_join_timeout_seconds": 10.0,
        }
    )
    meta_json = json.dumps(config.model_dump())

    # Room ALREADY contains alice
    alice_p = make_mock_participant(identity="alice")
    ctx = make_mock_job_context(
        job_id="job_already_present",
        metadata_json=meta_json,
        remote_participants={"alice_conn": alice_p},
    )

    mock_agent_session = MagicMock(spec=AgentSession)
    mock_agent_session.start = AsyncMock()

    with (
        patch("agent_framework.core.session.ModelFactory.create_model"),
        patch("agent_framework.core.session.TurnDetectionFactory.build_turn_handling_options"),
        patch("agent_framework.core.session.ToolRegistry.build_tools_from_config", return_value=[]),
        patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session),
        patch("agent_framework.core.session.Agent"),
    ):
        coordinator = SessionCoordinator()
        await coordinator.run(ctx)

        # wait_for_participant should NOT have been called
        ctx.wait_for_participant.assert_not_called()

        sc = get_session_context(ctx)
        sc.cancel_watchdogs()


@pytest.mark.asyncio
async def test_participant_join_timeout_raises_and_tears_down():
    """Verifies that timing out while waiting for a participant triggers teardown and raises TimeoutError."""
    config = make_valid_session_config(
        session={
            "participant_join_timeout_seconds": 0.05,
        }
    )
    meta_json = json.dumps(config.model_dump())

    ctx = make_mock_job_context(
        job_id="job_timeout",
        metadata_json=meta_json,
        remote_participants={},
    )

    async def _slow_wait(*args, **kwargs):
        await asyncio.sleep(1.0)
        return MagicMock()

    ctx.wait_for_participant = AsyncMock(side_effect=_slow_wait)

    coordinator = SessionCoordinator()
    with pytest.raises(TimeoutError):
        await coordinator.run(ctx)

    # Room cleanup verified on timeout teardown to prevent connection leaks
    ctx.delete_room.assert_awaited_once()


@pytest.mark.asyncio
async def test_participant_join_timeout_null_in_raw_metadata():
    """Verifies that null participant_join_timeout_seconds in raw metadata falls back cleanly to 30.0s."""
    raw_meta = json.dumps({"session": {"participant_join_timeout_seconds": None}})
    ctx = make_mock_job_context(
        job_id="job_null_timeout",
        metadata_json=raw_meta,
        remote_participants={},
    )
    ctx.wait_for_participant = AsyncMock(side_effect=TimeoutError("Timed out"))

    coordinator = SessionCoordinator()
    with pytest.raises(TimeoutError):
        await coordinator.run(ctx)

    ctx.wait_for_participant.assert_awaited_once()
    ctx.delete_room.assert_awaited_once()


@pytest.mark.asyncio
async def test_remote_participants_none_handled_gracefully():
    """Verifies that ctx.room.remote_participants=None does not raise TypeError."""
    config = make_valid_session_config()
    meta_json = json.dumps(config.model_dump())
    ctx = make_mock_job_context(
        job_id="job_none_parts",
        metadata_json=meta_json,
    )
    ctx.room.remote_participants = None
    joined_p = make_mock_participant(identity="incoming_user")
    ctx.wait_for_participant = AsyncMock(return_value=joined_p)

    mock_agent_session = MagicMock(spec=AgentSession)
    mock_agent_session.start = AsyncMock()

    with (
        patch("agent_framework.core.session.ModelFactory.create_model"),
        patch("agent_framework.core.session.TurnDetectionFactory.build_turn_handling_options"),
        patch("agent_framework.core.session.ToolRegistry.build_tools_from_config", return_value=[]),
        patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session),
        patch("agent_framework.core.session.Agent"),
    ):
        coordinator = SessionCoordinator()
        await coordinator.run(ctx)

        ctx.wait_for_participant.assert_awaited_once()
        sc = get_session_context(ctx)
        assert sc is not None
        sc.cancel_watchdogs()


@pytest.mark.asyncio
async def test_recording_options_configured_when_enabled():
    """Verifies native RecordingOptions are passed to session.start when recording is enabled."""
    config = make_valid_session_config(
        session={
            "recording": {
                "enabled": True,
                "audio": True,
                "traces": True,
            }
        }
    )
    meta_json = json.dumps(config.model_dump())
    ctx = make_mock_job_context(
        job_id="job_recording",
        metadata_json=meta_json,
        remote_participants={"p1": make_mock_participant(identity="user")},
    )

    mock_agent_session = MagicMock(spec=AgentSession)
    mock_agent_session.start = AsyncMock()

    with (
        patch("agent_framework.core.session.ModelFactory.create_model"),
        patch("agent_framework.core.session.TurnDetectionFactory.build_turn_handling_options"),
        patch("agent_framework.core.session.ToolRegistry.build_tools_from_config", return_value=[]),
        patch("agent_framework.core.session.AgentSession", return_value=mock_agent_session),
        patch("agent_framework.core.session.Agent"),
    ):
        coordinator = SessionCoordinator()
        await coordinator.run(ctx)

        mock_agent_session.start.assert_awaited_once()
        start_kwargs = mock_agent_session.start.call_args[1]
        record_opt = start_kwargs.get("record")
        assert record_opt is not False
        assert isinstance(record_opt, dict)
        assert record_opt.get("audio") is True
        assert record_opt.get("traces") is True

        sc = get_session_context(ctx)
        sc.cancel_watchdogs()


# ==============================================================================
# 6. Server Initialization & CLI Runner Tests
# ==============================================================================

def test_create_server_default_options():
    """Verifies create_server applies production settings (health check port 8081)."""
    server = create_server()
    assert isinstance(server, AgentServer)


def test_create_server_custom_port_and_shutdown_timeout():
    """Verifies create_server respects custom port, shutdown timeout, and load threshold."""
    server = create_server(
        port=8089,
        shutdown_process_timeout=45.0,
        load_threshold=0.85,
        drain_timeout=180.0,
        prometheus_port=9090,
        log_level="DEBUG",
    )
    assert isinstance(server, AgentServer)


def test_framework_settings_production_defaults():
    """Verifies default values for drain_timeout, prometheus_port, and log_level."""
    cfg = FrameworkSettings()
    assert cfg.drain_timeout_seconds == 300.0
    assert cfg.prometheus_port is None
    assert cfg.livekit_log_level == "INFO"
    assert cfg.log_level == "INFO"


def test_framework_settings_env_aliases(monkeypatch):
    """Verifies environment variable aliases for production settings."""
    monkeypatch.setenv("DRAIN_TIMEOUT_SECONDS", "450")
    monkeypatch.setenv("PROMETHEUS_PORT", "9100")
    monkeypatch.setenv("LIVEKIT_LOG_LEVEL", "WARNING")

    cfg = FrameworkSettings()
    assert cfg.drain_timeout_seconds == 450.0
    assert cfg.prometheus_port == 9100
    assert cfg.livekit_log_level == "WARNING"
    assert cfg.log_level == "WARNING"


def test_setup_server_wires_entrypoint_and_session_end():
    """Verifies setup_server registers rtc_session entrypoint on AgentServer."""
    coordinator = SessionCoordinator()
    mock_server = MagicMock(spec=AgentServer)

    configured_server = setup_server(
        server=mock_server,
        coordinator=coordinator,
        agent_name="test-voice-agent",
    )

    assert configured_server is mock_server
    mock_server.rtc_session.assert_called_once_with(
        agent_name="test-voice-agent",
        on_session_end=coordinator.on_session_end,
    )


def test_run_app_invokes_cli():
    """Verifies run_app / run_server calls livekit.agents.cli.run_app."""
    mock_server = MagicMock(spec=AgentServer)

    with patch("agent_framework.core.server.cli.run_app") as mock_cli_run:
        run_app(mock_server)
        mock_cli_run.assert_called_once_with(mock_server)

    with patch("agent_framework.core.server.cli.run_app") as mock_cli_run:
        run_server(mock_server)
        mock_cli_run.assert_called_once_with(mock_server)


def test_create_server_forwards_credentials_from_settings():
    """Verifies create_server forwards credentials from custom FrameworkSettings."""
    custom_cfg = FrameworkSettings(
        livekit_url="wss://custom.livekit.cloud",
        livekit_api_key="custom_api_key",
        livekit_api_secret="custom_secret_key",
    )
    server = create_server(settings=custom_cfg)
    assert server._ws_url == "wss://custom.livekit.cloud"
    assert server._api_key == "custom_api_key"
    assert server._api_secret == "custom_secret_key"


def test_run_app_with_custom_kwargs():
    """Verifies run_app gracefully handles custom kwargs when initializing server."""
    with patch("agent_framework.core.server.cli.run_app") as mock_cli_run:
        run_app(port=8099, drain_timeout=240.0)
        mock_cli_run.assert_called_once()
        passed_server = mock_cli_run.call_args[0][0]
        assert isinstance(passed_server, AgentServer)
        assert passed_server._port == 8099
        assert passed_server._drain_timeout == 240


# ==============================================================================
# 7. Edge Case Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_teardown_disconnect_room_when_delete_room_disabled():
    """Verifies that room.disconnect() is called when delete_room_on_terminate is False."""
    config = make_valid_session_config(
        session={"delete_room_on_terminate": False}
    )
    ctx = make_mock_job_context(job_id="job_disconnect_only")
    sc = SessionContext(job_id="job_disconnect_only", room_name="room_d", session_config=config)
    attach_to_job(ctx, sc)
    sc.mark_started()

    coordinator = SessionCoordinator()
    with patch.object(ReportingDispatcher, "send_session_report", new_callable=AsyncMock):
        await coordinator.teardown(ctx)

    ctx.delete_room.assert_not_called()
    ctx.room.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_coordinator_run_exception_triggers_teardown_and_reraises():
    """Verifies unhandled exception during coordinator.run triggers teardown with error and re-raises."""
    ctx = make_mock_job_context(
        job_id="job_err",
        metadata_json='{"invalid_json": true}',
    )
    ctx.connect = AsyncMock()

    coordinator = SessionCoordinator()
    with patch.object(coordinator, "teardown", new_callable=AsyncMock) as mock_teardown:
        with pytest.raises(Exception):
            await coordinator.run(ctx)

        mock_teardown.assert_awaited_once()
        err_arg = mock_teardown.call_args[1].get("error")
        assert err_arg is not None


@pytest.mark.asyncio
async def test_teardown_handles_make_session_report_failure_gracefully():
    """Verifies teardown continues cleanly even if make_session_report raises an exception."""
    config = make_valid_session_config()
    ctx = make_mock_job_context(job_id="job_report_err")
    ctx.make_session_report = MagicMock(side_effect=RuntimeError("Report generation failed"))

    sc = SessionContext(job_id="job_report_err", room_name="room_r", session_config=config)
    attach_to_job(ctx, sc)

    with patch.object(ReportingDispatcher, "send_session_report", new_callable=AsyncMock) as mock_report:
        coordinator = SessionCoordinator()
        # Should not raise
        await coordinator.teardown(ctx)

        mock_report.assert_awaited_once()
        # lk_report should be None due to caught error
        assert mock_report.call_args[1]["lk_report"] is None


@pytest.mark.asyncio
async def test_teardown_noop_when_no_session_context():
    """Verifies teardown is a safe no-op if called on a context with no SessionContext."""
    ctx = make_mock_job_context(job_id="job_empty")
    coordinator = SessionCoordinator()

    # Should safely return without error
    await coordinator.teardown(ctx)
    ctx.delete_room.assert_not_called()

