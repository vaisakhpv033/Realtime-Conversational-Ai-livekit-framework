"""Typed SessionContext and scoped lifecycle state management for LiveKit Agent jobs.

Eliminates module-level global stores (such as _JOB_METADATA_STORE) by scoping
all session lifecycle data directly to the active LiveKit JobContext.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from typing import Any, Optional

from livekit.agents import Agent, AgentSession, JobContext

from agent_framework.config.schema import SessionConfig

logger = logging.getLogger("agent_framework.core.context")

# Context attribute key for JobContext instance storage
_SESSION_CONTEXT_ATTR = "_agent_framework_session_context"
_USERDATA_KEY_PREFIX = "_agent_framework_session_context:"


@dataclass
class SessionContext:
    """Encapsulates per-job state and lifecycle tracking bound to a JobContext.

    Tracks active configuration, LiveKit Agent and AgentSession instances,
    runtime timestamps, watchdog background tasks, and post-session telemetry state.
    """

    job_id: str
    room_name: str
    session_config: Optional[SessionConfig] = None
    model: Optional[Any] = None
    agent: Optional[Agent] = None
    session: Optional[AgentSession] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: str = "initialized"
    error: Optional[Exception] = None
    termination_reason: Optional[str] = None
    max_duration_task: Optional[asyncio.Task[Any]] = None
    inactivity_task: Optional[asyncio.Task[Any]] = None
    reporting_sent: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_data: dict[str, Any] = field(default_factory=dict)

    def mark_started(self) -> None:
        """Marks the session as actively running."""
        if self.started_at is None:
            self.started_at = datetime.now(UTC)
        self.status = "running"
        logger.debug(f"SessionContext [{self.job_id}] marked started at {self.started_at.isoformat()}")

    def mark_ended(
        self,
        status: str = "completed",
        error: Optional[Exception] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Marks the session as concluded with final status and reason."""
        if self.ended_at is None:
            self.ended_at = datetime.now(UTC)
        self.status = status
        if error is not None:
            self.error = error
        if reason is not None:
            self.termination_reason = reason
        logger.debug(
            f"SessionContext [{self.job_id}] marked ended at {self.ended_at.isoformat()} "
            f"(status={status}, reason={reason})"
        )

    @property
    def duration_seconds(self) -> float:
        """Returns the total elapsed duration in seconds."""
        if self.started_at is None:
            return 0.0
        end_time = self.ended_at or datetime.now(UTC)
        return max(0.0, (end_time - self.started_at).total_seconds())

    @property
    def is_active(self) -> bool:
        """Returns True if the session is currently active."""
        return self.status == "running"

    def cancel_watchdogs(self) -> None:
        """Safely cancels active watchdog background tasks."""
        if self.max_duration_task is not None and not self.max_duration_task.done():
            self.max_duration_task.cancel()
            self.max_duration_task = None
            logger.debug(f"SessionContext [{self.job_id}]: max_duration watchdog cancelled.")

        if self.inactivity_task is not None and not self.inactivity_task.done():
            self.inactivity_task.cancel()
            self.inactivity_task = None
            logger.debug(f"SessionContext [{self.job_id}]: inactivity watchdog cancelled.")


def attach_to_job(ctx: JobContext, session_context: SessionContext) -> None:
    """Attaches a typed SessionContext to a LiveKit JobContext.

    Stores the context directly on the JobContext object and inside
    ctx.proc.userdata (keyed strictly by job_id) to guarantee session isolation without
    any global process state.

    Args:
        ctx: Active LiveKit JobContext.
        session_context: SessionContext instance to bind.
    """
    if ctx is None:
        raise ValueError("Cannot attach SessionContext to None ctx")

    # 1. Attach directly to JobContext instance
    try:
        setattr(ctx, _SESSION_CONTEXT_ATTR, session_context)
    except Exception as e:
        logger.warning(f"Could not set direct attribute on ctx: {e}")

    # 2. Attach to ctx.proc.userdata strictly keyed by job_id to guarantee multi-tenant isolation
    proc = getattr(ctx, "proc", None)
    if proc is not None and hasattr(proc, "userdata") and isinstance(proc.userdata, dict):
        if session_context.job_id:
            key = f"{_USERDATA_KEY_PREFIX}{session_context.job_id}"
            proc.userdata[key] = session_context

    logger.debug(f"Attached SessionContext [{session_context.job_id}] to JobContext.")


def get_session_context(ctx: Optional[JobContext]) -> Optional[SessionContext]:
    """Retrieves the SessionContext bound to the given JobContext.

    Args:
        ctx: Active LiveKit JobContext.

    Returns:
        The bound SessionContext if present, or None.
    """
    if ctx is None:
        return None

    # 1. Check direct attribute on JobContext
    direct = getattr(ctx, _SESSION_CONTEXT_ATTR, None)
    if direct is not None and isinstance(direct, SessionContext):
        return direct

    # 2. Check ctx.proc.userdata strictly keyed by job.id
    proc = getattr(ctx, "proc", None)
    if proc is not None and hasattr(proc, "userdata") and isinstance(proc.userdata, dict):
        job_id = getattr(ctx.job, "id", None) if getattr(ctx, "job", None) else None
        if job_id:
            key = f"{_USERDATA_KEY_PREFIX}{job_id}"
            val = proc.userdata.get(key)
            if val is not None and isinstance(val, SessionContext):
                return val

    # 3. Check dictionary-style ctx for lightweight mocks / tests
    if isinstance(ctx, dict):
        val = ctx.get(_SESSION_CONTEXT_ATTR) or ctx.get("session_context")
        if val is not None and isinstance(val, SessionContext):
            return val

    return None


def detach_from_job(ctx: Optional[JobContext]) -> Optional[SessionContext]:
    """Detaches and cleans up the SessionContext from a JobContext.

    Args:
        ctx: Active LiveKit JobContext.

    Returns:
        The detached SessionContext, or None if none was attached.
    """
    if ctx is None:
        return None

    session_context = get_session_context(ctx)

    # 1. Remove direct attribute
    if hasattr(ctx, _SESSION_CONTEXT_ATTR):
        try:
            delattr(ctx, _SESSION_CONTEXT_ATTR)
        except AttributeError:
            pass

    # 2. Remove strictly keyed job context from ctx.proc.userdata
    proc = getattr(ctx, "proc", None)
    if proc is not None and hasattr(proc, "userdata") and isinstance(proc.userdata, dict):
        job_id = (
            session_context.job_id
            if session_context
            else (getattr(ctx.job, "id", None) if getattr(ctx, "job", None) else None)
        )
        if job_id:
            proc.userdata.pop(f"{_USERDATA_KEY_PREFIX}{job_id}", None)

    if session_context is not None:
        logger.debug(f"Detached SessionContext [{session_context.job_id}] from JobContext.")

    return session_context
