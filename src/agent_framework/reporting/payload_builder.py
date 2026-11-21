"""Application-agnostic session telemetry and reporting payload builder.

Constructs standardized, domain-neutral reporting payloads containing:
- Identification: session_id, job_id, room_name, agent identity
- Timestamps and duration: started_at, ended_at, duration_seconds
- Termination: termination_reason, error diagnostics (if any)
- Conversational telemetry: chat_history turns, model_usage metrics, timeline events
- Audio artifacts: audio_recording_path (if recording enabled)
- Pass-through context: custom_fields and user metadata

Zero domain-specific rubrics, trainee labels, or clinical assumptions.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import Any, Optional

from agent_framework.config.schema import SessionConfig

logger = logging.getLogger("agent_framework.reporting.payload_builder")


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Safely retrieves a field from an object or dictionary."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_chat_history(chat_history_raw: Any) -> list[dict[str, Any]] | None:
    """Extracts a normalized list of chat turns from LiveKit ChatContext, dict, or list.

    Returns turns in standard format:
        [{"role": "user", "content": "text", "timestamp": float | str | None}, ...]
    """
    if chat_history_raw is None:
        return None

    items: list[Any] = []
    if isinstance(chat_history_raw, list):
        items = chat_history_raw
    elif isinstance(chat_history_raw, dict):
        if "items" in chat_history_raw and isinstance(chat_history_raw["items"], list):
            items = chat_history_raw["items"]
        elif "messages" in chat_history_raw and isinstance(chat_history_raw["messages"], list):
            items = chat_history_raw["messages"]
        else:
            items = [chat_history_raw]
    elif hasattr(chat_history_raw, "items") and not callable(chat_history_raw.items):
        # LiveKit ChatContext.items property
        items = list(chat_history_raw.items)
    elif hasattr(chat_history_raw, "messages") and callable(chat_history_raw.messages):
        try:
            items = list(chat_history_raw.messages())
        except Exception:
            items = []
    elif hasattr(chat_history_raw, "to_dict"):
        try:
            d = chat_history_raw.to_dict(exclude_timestamp=False)
            if isinstance(d, dict) and "items" in d:
                items = d["items"]
            elif isinstance(d, list):
                items = d
        except Exception as e:
            logger.debug(f"Error calling to_dict on chat_history: {e}")

    turns: list[dict[str, Any]] = []
    for item in items:
        role: str = "user"
        content: Any = ""
        timestamp: Any = None

        if isinstance(item, str):
            content = item
            timestamp = None
        elif isinstance(item, dict):
            raw_role = item.get("role", "user")
            content = item.get("content", "")
            timestamp = item.get("timestamp") or item.get("created_at")
            if hasattr(raw_role, "value"):
                role = str(raw_role.value).lower()
            else:
                role = str(raw_role).lower()
        else:
            # Check for LiveKit FunctionCall / FunctionCallOutput items
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                role = "assistant"
                fn_name = getattr(item, "name", "tool")
                fn_args = getattr(item, "arguments", "")
                content = f"Tool call: {fn_name}({fn_args})"
                timestamp = getattr(item, "created_at", None) or getattr(item, "timestamp", None)
            elif item_type == "function_call_output":
                role = "tool"
                content = str(getattr(item, "output", ""))
                timestamp = getattr(item, "created_at", None) or getattr(item, "timestamp", None)
            else:
                raw_role = getattr(item, "role", "user")
                if hasattr(raw_role, "value"):
                    role = str(raw_role.value).lower()
                else:
                    role = str(raw_role).lower()

                # Prefer text_content property on LiveKit ChatMessage (strips markup tags)
                content = getattr(item, "text_content", None)
                if content is None:
                    content = getattr(item, "content", "")

                timestamp = getattr(item, "timestamp", None) or getattr(item, "created_at", None)

        # Format content string cleanly
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif hasattr(part, "text"):
                    text_parts.append(str(part.text))
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(str(part["text"]))
                else:
                    text_parts.append(str(part))
            content_str = " ".join(text_parts).strip()
        else:
            content_str = str(content).strip() if content is not None else ""

        turns.append({
            "role": role,
            "content": content_str,
            "timestamp": timestamp,
        })

    return turns


def _extract_model_usage(model_usage_raw: Any) -> list[dict[str, Any]] | None:
    """Extracts token counts and cost metrics into a standardized list of dicts."""
    if model_usage_raw is None:
        return None

    if hasattr(model_usage_raw, "model_usage"):
        model_usage_raw = model_usage_raw.model_usage

    if isinstance(model_usage_raw, list):
        formatted: list[dict[str, Any]] = []
        for item in model_usage_raw:
            if hasattr(item, "model_dump"):
                formatted.append(item.model_dump(mode="json"))
            elif hasattr(item, "to_dict"):
                formatted.append(item.to_dict())
            elif isinstance(item, dict):
                formatted.append(dict(item))
            elif hasattr(item, "__dict__"):
                formatted.append(dict(vars(item)))
            else:
                formatted.append({"usage": str(item)})
        return formatted

    if isinstance(model_usage_raw, dict):
        return [dict(model_usage_raw)]

    if hasattr(model_usage_raw, "model_dump"):
        return [model_usage_raw.model_dump(mode="json")]

    return None


def _extract_events(events_raw: Any) -> list[dict[str, Any]] | None:
    """Extracts timeline events from SessionReport events list."""
    if events_raw is None:
        return None

    if isinstance(events_raw, list):
        formatted: list[dict[str, Any]] = []
        for event in events_raw:
            if hasattr(event, "model_dump"):
                formatted.append(event.model_dump(mode="json"))
            elif hasattr(event, "to_dict"):
                formatted.append(event.to_dict())
            elif isinstance(event, dict):
                formatted.append(dict(event))
            elif hasattr(event, "__dict__"):
                formatted.append(dict(vars(event)))
            else:
                formatted.append({"event": str(event)})
        return formatted

    return None


def build_session_telemetry_payload(
    ctx: Any,
    session_config: SessionConfig,
    lk_report: Any = None,
    error: Exception | None = None,
    termination_reason: str | None = None,
) -> dict[str, Any]:
    """Assembles application-agnostic session telemetry payload for post-session reporting.

    Resolution logic:
    - session_id: metadata['session_id'] -> ctx.job.id -> session_config.agent.name
    - job_id: ctx.job.id -> lk_report.job_id -> 'unknown'
    - room_name: ctx.room.name -> ctx.job.room.name -> lk_report.room -> 'unknown'
    - agent: {'name': session_config.agent.name, 'role': session_config.agent.role}
    - chat_history: extracted from lk_report or ctx.make_session_report() if include_chat_history=True
    - model_usage: extracted from lk_report if include_model_usage=True
    - events: extracted from lk_report if include_events=True
    - audio_recording_path: from lk_report.audio_recording_path
    - custom_fields: session_config.reporting.payload_config.custom_fields
    - metadata: session_config.metadata pass-through

    Args:
        ctx: LiveKit JobContext or mock context.
        session_config: Validated SessionConfig metadata contract.
        lk_report: Optional SessionReport object or dict.
        error: Optional Exception that caused session termination.
        termination_reason: Optional explicit reason ('normal', 'user_hangup', 'timeout', etc.).

    Returns:
        Standardized telemetry payload dictionary.
    """
    # Attempt to extract LiveKit SessionReport if not provided
    if lk_report is None and ctx is not None and hasattr(ctx, "make_session_report"):
        try:
            lk_report = ctx.make_session_report()
        except Exception as e:
            logger.debug(f"Could not extract standard LiveKit session report: {e}")

    # Identifiers
    metadata_dict = session_config.metadata or {}
    ctx_job = getattr(ctx, "job", None) if ctx else None
    ctx_room = getattr(ctx, "room", None) if ctx else None
    ctx_job_room = getattr(ctx_job, "room", None) if ctx_job else None

    job_id = (
        _get_field(ctx_job, "id")
        or _get_field(lk_report, "job_id")
        or "unknown"
    )
    room_name = (
        _get_field(ctx_room, "name")
        or _get_field(ctx_job_room, "name")
        or _get_field(lk_report, "room")
        or _get_field(lk_report, "room_name")
        or "unknown"
    )
    session_id = (
        metadata_dict.get("session_id")
        or (job_id if job_id != "unknown" else None)
        or session_config.agent.name
        or "unknown"
    )

    # Agent identity
    agent_info = {
        "name": session_config.agent.name,
        "role": session_config.agent.role,
    }

    # Timestamps & Duration
    ended_dt = datetime.now(UTC)
    ended_at_str = ended_dt.isoformat().replace("+00:00", "Z")
    started_at_str: Optional[str] = None
    duration_seconds: Optional[float] = None

    report_started = _get_field(lk_report, "started_at")
    if report_started is None and ctx is not None:
        primary_session = getattr(ctx, "_primary_agent_session", None) or getattr(ctx, "session", None)
        if primary_session is not None:
            report_started = getattr(primary_session, "_started_at", None)

    if report_started is not None:
        if isinstance(report_started, (int, float)):
            started_dt = datetime.fromtimestamp(report_started, UTC)
            started_at_str = started_dt.isoformat().replace("+00:00", "Z")
            duration_seconds = max(0.0, (ended_dt - started_dt).total_seconds())
        elif isinstance(report_started, datetime):
            started_at_str = report_started.isoformat().replace("+00:00", "Z")
            duration_seconds = max(0.0, (ended_dt - report_started).total_seconds())
        elif isinstance(report_started, str):
            started_at_str = report_started

    report_duration = _get_field(lk_report, "duration")
    if report_duration is not None:
        try:
            duration_seconds = float(report_duration)
        except (ValueError, TypeError):
            pass

    # Termination reason
    if termination_reason:
        term_reason = termination_reason
    elif error is not None:
        term_reason = "error"
    else:
        term_reason = "normal"

    payload_cfg = session_config.reporting.payload_config

    # Chat history
    chat_history: Optional[list[dict[str, Any]]] = None
    if payload_cfg.include_chat_history:
        raw_history = _get_field(lk_report, "chat_history")
        chat_history = _extract_chat_history(raw_history)

    # Model usage metrics
    model_usage: Optional[list[dict[str, Any]]] = None
    if payload_cfg.include_model_usage:
        raw_usage = _get_field(lk_report, "model_usage") or _get_field(lk_report, "usage")
        model_usage = _extract_model_usage(raw_usage)

    # Events timeline
    events: Optional[list[dict[str, Any]]] = None
    if payload_cfg.include_events:
        raw_events = _get_field(lk_report, "events")
        events = _extract_events(raw_events)

    # Audio recording path
    audio_path = _get_field(lk_report, "audio_recording_path")
    audio_recording_path: Optional[str] = None
    if audio_path is not None:
        audio_recording_path = str(audio_path.absolute()) if hasattr(audio_path, "absolute") else str(audio_path)

    # Pass-through context
    custom_fields = dict(payload_cfg.custom_fields)
    metadata_passthrough = dict(metadata_dict)

    payload: dict[str, Any] = {
        "session_id": str(session_id),
        "job_id": str(job_id),
        "room_name": str(room_name),
        "agent": agent_info,
        "started_at": started_at_str,
        "ended_at": ended_at_str,
        "duration_seconds": duration_seconds,
        "termination_reason": term_reason,
        "chat_history": chat_history,
        "model_usage": model_usage,
        "events": events,
        "audio_recording_path": audio_recording_path,
        "custom_fields": custom_fields,
        "metadata": metadata_passthrough,
    }

    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }

    return payload

