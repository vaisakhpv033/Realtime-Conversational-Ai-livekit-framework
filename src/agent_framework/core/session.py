"""Unified Session Coordinator for LiveKit Voice Agent Framework.

Coordinates the end-to-end lifecycle of an agent session:
1. Room connection via ctx.connect()
2. Participant join waiting with identity targeting and timeout
3. Cascading metadata extraction and strict schema validation
4. Dynamic model adapter creation via ModelFactory
5. Turn handling and speech dynamics via TurnDetectionFactory
6. Function tool resolution via ToolRegistry
7. Effective system instructions preparation
8. Agent and AgentSession instantiation
9. Recording configuration management
10. First-turn greeting execution (say, generate_reply, or passive listen)
11. Max duration and inactivity watchdog timers
12. Resilient teardown and post-call reporting via ReportingDispatcher
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import inspect
import logging
from pathlib import Path
from typing import Any, Optional, Union

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RecordingOptions,
)

from agent_framework.config.resolver import (
    extract_session_metadata,
    resolve_cascading_metadata,
)
from agent_framework.config.schema import (
    InstructionsConfig,
    SessionConfig,
    get_effective_system_prompt,
)
from agent_framework.config.validator import validate_metadata
from agent_framework.core.context import (
    SessionContext,
    attach_to_job,
    detach_from_job,
    get_session_context,
)
from agent_framework.models.factory import ModelFactory
from agent_framework.reporting.dispatcher import ReportingDispatcher
from agent_framework.tools.registry import ToolRegistry
from agent_framework.turn_detection.factory import TurnDetectionFactory

logger = logging.getLogger("agent_framework.core.session")


class SessionCoordinator:
    """Coordinates the unified, application-agnostic LiveKit voice agent session lifecycle."""

    def __init__(
        self,
        *,
        model_factory: type[ModelFactory] = ModelFactory,
        turn_detection_factory: type[TurnDetectionFactory] = TurnDetectionFactory,
        tool_registry: type[ToolRegistry] = ToolRegistry,
        reporting_dispatcher: type[ReportingDispatcher] = ReportingDispatcher,
        dead_letter_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes the session coordinator with pluggable factories.

        Args:
            model_factory: Factory class for model creation.
            turn_detection_factory: Factory class for turn-taking options.
            tool_registry: Tool registry class for function tools.
            reporting_dispatcher: Dispatcher class for post-session telemetry.
            dead_letter_dir: Optional custom directory for dead-letter storage.
        """
        self.model_factory = model_factory
        self.turn_detection_factory = turn_detection_factory
        self.tool_registry = tool_registry
        self.reporting_dispatcher = reporting_dispatcher
        self.dead_letter_dir = dead_letter_dir

    async def run(self, ctx: JobContext) -> None:
        """Executes the complete session pipeline for a LiveKit JobContext.

        Args:
            ctx: Active LiveKit JobContext dispatched for the room connection.
        """
        job_id = getattr(ctx.job, "id", "unknown") if getattr(ctx, "job", None) else "unknown"
        room_name = getattr(ctx.room, "name", "unknown") if getattr(ctx, "room", None) else "unknown"
        logger.info(f"SessionCoordinator starting for job '{job_id}' in room '{room_name}'...")

        # Initialize and attach SessionContext early to guarantee lifecycle tracking and teardown safety
        session_context = SessionContext(
            job_id=job_id,
            room_name=room_name,
        )
        attach_to_job(ctx, session_context)

        try:
            # 1. Connect to the room
            if hasattr(ctx, "connect"):
                await ctx.connect()
                logger.info(f"Connected to room '{room_name}' (job: {job_id})")

            # 2. Handle participant waiting
            # Determine join timeout and target identity from preliminary metadata
            target_identity = None
            join_timeout = 30.0

            try:
                prelim_config = extract_session_metadata(ctx)
                join_timeout = prelim_config.session.participant_join_timeout_seconds
                target_identity = prelim_config.session.target_participant_identity
                session_context.session_config = prelim_config
            except Exception:
                # If strict validation fails prior to participant joining, inspect raw metadata
                merged_raw = resolve_cascading_metadata(
                    getattr(ctx.job, "metadata", None) if getattr(ctx, "job", None) else None,
                    getattr(ctx.room, "metadata", None) if getattr(ctx, "room", None) else None,
                )
                sess_raw = merged_raw.get("session", {})
                if isinstance(sess_raw, dict):
                    raw_timeout = sess_raw.get("participant_join_timeout_seconds")
                    if raw_timeout is not None:
                        try:
                            join_timeout = float(raw_timeout)
                        except (ValueError, TypeError):
                            join_timeout = 30.0
                    else:
                        join_timeout = 30.0
                    target_identity = sess_raw.get("target_participant_identity")

            remote_parts = (
                (getattr(ctx.room, "remote_participants", None) or {})
                if getattr(ctx, "room", None)
                else {}
            )

            target_already_present = False
            if target_identity and remote_parts:
                target_already_present = any(
                    getattr(p, "identity", None) == target_identity for p in remote_parts.values()
                )

            needs_wait = False
            if join_timeout and join_timeout > 0:
                if target_identity:
                    needs_wait = not target_already_present
                else:
                    needs_wait = len(remote_parts) == 0

            if needs_wait and hasattr(ctx, "wait_for_participant"):
                logger.info(
                    f"Waiting for participant (identity={target_identity}, timeout={join_timeout}s)..."
                )
                joined_p = await asyncio.wait_for(
                    ctx.wait_for_participant(identity=target_identity),
                    timeout=float(join_timeout),
                )
                if not target_identity and joined_p is not None:
                    target_identity = getattr(joined_p, "identity", None)
                logger.info(f"Participant joined room '{room_name}' for job '{job_id}'")

            # 3. Resolve and validate metadata via extract_session_metadata(ctx) and validate_metadata()
            session_config = extract_session_metadata(ctx, target_participant_identity=target_identity)
            session_config = validate_metadata(session_config)
            session_context.session_config = session_config

            # Register shutdown callback on JobContext if available
            if hasattr(ctx, "add_shutdown_callback"):
                async def _on_job_shutdown(reason: str = "") -> None:
                    await self.teardown(ctx, termination_reason=reason or None)

                ctx.add_shutdown_callback(_on_job_shutdown)

            # 4. Instantiate model via ModelFactory.create_model(session_config)
            llm = self.model_factory.create_model(session_config)
            session_context.model = llm

            # 5. Instantiate turn-taking options via TurnDetectionFactory
            turn_handling = self.turn_detection_factory.build_turn_handling_options(session_config)
            user_away_timeout = self.turn_detection_factory.get_user_away_timeout(session_config)

            # 6. Instantiate dynamic tools via ToolRegistry.build_tools_from_config(session_config, ctx=ctx)
            tools = self.tool_registry.build_tools_from_config(session_config, ctx=ctx)

            # 7. Prepare effective system instructions
            effective_instructions = get_effective_system_prompt(session_config.instructions)

            # 8. Construct livekit.agents.Agent
            agent = Agent(
                instructions=effective_instructions,
                tools=tools,
            )
            session_context.agent = agent

            # 9. Construct livekit.agents.AgentSession
            session = AgentSession(
                llm=llm,
                turn_handling=turn_handling,
                user_away_timeout=user_away_timeout,
                tools=tools,
            )
            session_context.session = session

            # 10. Manage native recording
            record_param: Any = False
            if session_config.session.recording.enabled:
                record_param = RecordingOptions(
                    audio=session_config.session.recording.audio,
                    traces=session_config.session.recording.traces,
                )

            # Start AgentSession
            await session.start(
                agent=agent,
                room=getattr(ctx, "room", None),
                record=record_param,
            )
            session_context.mark_started()
            logger.info(f"AgentSession successfully started for job '{job_id}'")

            # 11. Set up session watchdogs
            self._setup_watchdogs(ctx, session_context)

            # 12. Execute first-turn greeting
            first_turn = session_config.instructions.first_turn
            if first_turn.speaker == "agent":
                if first_turn.mode == "say" and first_turn.text:
                    activity = getattr(session, "_activity", None)
                    tts_available = bool(
                        getattr(session, "tts", None)
                        or (activity and getattr(activity, "tts", None))
                    )
                    active_llm = llm or getattr(session, "llm", None)
                    supports_say = bool(
                        getattr(getattr(active_llm, "capabilities", None), "supports_say", False)
                    )

                    if tts_available or supports_say:
                        logger.info(f"Executing agent first-turn say: '{first_turn.text[:40]}...'")
                        await session.say(first_turn.text)
                    else:
                        logger.info(
                            f"Model does not support direct say(); routing first-turn greeting via generate_reply: '{first_turn.text[:40]}...'"
                        )
                        reply_handle = session.generate_reply(
                            instructions=f"Please greet the user by saying: '{first_turn.text}'"
                        )
                        if inspect.isawaitable(reply_handle):
                            await reply_handle
                elif first_turn.mode == "generate_reply":
                    logger.info("Executing agent first-turn generate_reply...")
                    reply_handle = session.generate_reply()
                    if inspect.isawaitable(reply_handle):
                        await reply_handle
            else:
                logger.debug("First turn speaker is 'user'; agent listening passively.")

        except Exception as e:
            logger.error(f"Error in SessionCoordinator for job '{job_id}': {e}", exc_info=True)
            await self.teardown(ctx, error=e)
            raise

    def _setup_watchdogs(self, ctx: JobContext, session_context: SessionContext) -> None:
        """Configures max-duration and inactivity watchdog tasks for the active session."""
        session_config = session_context.session_config
        coordinator = self

        # Watchdog 1: Max call duration
        max_duration = session_config.session.max_duration_seconds
        if max_duration and max_duration > 0:

            async def _max_duration_watchdog() -> None:
                try:
                    await asyncio.sleep(max_duration)
                    if not session_context.is_active or session_context.reporting_sent:
                        return
                    logger.warning(
                        f"Session [{session_context.job_id}] reached max duration of {max_duration}s. Terminating..."
                    )
                    session_context.mark_ended(
                        status="timed_out",
                        reason=f"Max duration of {max_duration}s exceeded",
                    )
                    await coordinator.teardown(
                        ctx,
                        termination_reason=f"Max duration of {max_duration}s exceeded",
                    )
                except asyncio.CancelledError:
                    pass
                except Exception as ex:
                    logger.error(f"Error in max duration watchdog: {ex}", exc_info=True)

            session_context.max_duration_task = asyncio.create_task(
                _max_duration_watchdog(),
                name=f"max_duration_{session_context.job_id}",
            )

        # Watchdog 2: Inactivity timeout
        inactivity_timeout = session_config.session.inactivity_timeout_seconds
        if inactivity_timeout and inactivity_timeout > 0:
            loop = asyncio.get_running_loop()
            last_activity_time = loop.time()

            def _on_activity(*args: Any, **kwargs: Any) -> None:
                nonlocal last_activity_time
                try:
                    last_activity_time = asyncio.get_running_loop().time()
                except RuntimeError:
                    pass

            session = session_context.session
            if session is not None and hasattr(session, "on"):
                for evt in (
                    "user_state_changed",
                    "agent_state_changed",
                    "conversation_item_added",
                    "user_input_transcribed",
                    "tool_execution_updated",
                ):
                    try:
                        session.on(evt, _on_activity)
                    except Exception:
                        pass

            async def _inactivity_watchdog() -> None:
                nonlocal last_activity_time
                try:
                    while True:
                        check_interval = min(5.0, max(0.5, inactivity_timeout / 4.0))
                        await asyncio.sleep(check_interval)
                        if not session_context.is_active or session_context.reporting_sent:
                            break
                        now = asyncio.get_running_loop().time()
                        if (now - last_activity_time) >= inactivity_timeout:
                            logger.warning(
                                f"Session [{session_context.job_id}] idle for {inactivity_timeout}s. Terminating..."
                            )
                            session_context.mark_ended(
                                status="timed_out",
                                reason=f"Inactivity timeout of {inactivity_timeout}s exceeded",
                            )
                            await coordinator.teardown(
                                ctx,
                                termination_reason=f"Inactivity timeout of {inactivity_timeout}s exceeded",
                            )
                            break
                except asyncio.CancelledError:
                    pass
                except Exception as ex:
                    logger.error(f"Error in inactivity watchdog: {ex}", exc_info=True)

            session_context.inactivity_task = asyncio.create_task(
                _inactivity_watchdog(),
                name=f"inactivity_{session_context.job_id}",
            )

    async def teardown(
        self,
        ctx: JobContext,
        error: Optional[Exception] = None,
        termination_reason: Optional[str] = None,
    ) -> None:
        """Manages teardown and post-call reporting on disconnect or error.

        Guaranteed idempotent per session: only dispatches telemetry and cleans
        up resources once.

        Args:
            ctx: Active LiveKit JobContext.
            error: Optional Exception that caused session termination.
            termination_reason: Optional human-readable termination reason.
        """
        session_context = get_session_context(ctx)
        if session_context is None:
            logger.debug("Teardown invoked but no SessionContext attached to JobContext.")
            return

        # Ensure idempotency
        if session_context.reporting_sent:
            logger.debug(f"Teardown/reporting already processed for job '{session_context.job_id}'.")
            return
        session_context.reporting_sent = True

        logger.info(f"Tearing down session for job '{session_context.job_id}'...")

        # 1. Cancel watchdogs
        session_context.cancel_watchdogs()

        # 2. Mark session ended
        final_status = "failed" if error else (
            session_context.status if session_context.status != "running" else "completed"
        )
        session_context.mark_ended(
            status=final_status,
            error=error,
            reason=termination_reason or session_context.termination_reason,
        )

        # 3. Generate LiveKit session report if available
        lk_report = None
        try:
            if hasattr(ctx, "make_session_report"):
                lk_report = ctx.make_session_report(session_context.session)
        except Exception as ex:
            logger.warning(f"Could not generate LiveKit session report: {ex}")

        # 4. Trigger post-session reporting
        try:
            if session_context.session_config and session_context.session_config.reporting.enabled:
                report_kwargs: dict[str, Any] = {}
                if self.dead_letter_dir is not None:
                    report_kwargs["dead_letter_dir"] = self.dead_letter_dir
                await self.reporting_dispatcher.send_session_report(
                    ctx=ctx,
                    session_config=session_context.session_config,
                    lk_report=lk_report,
                    error=error,
                    termination_reason=termination_reason or session_context.termination_reason,
                    **report_kwargs,
                )
        except Exception as ex:
            logger.error(f"Error sending session report: {ex}", exc_info=True)

        # 5. Clean up AgentSession
        try:
            if session_context.session and hasattr(session_context.session, "aclose"):
                await session_context.session.aclose()
        except Exception as ex:
            logger.warning(f"Error closing agent session: {ex}")

        # 6. Clean up room resources
        try:
            should_delete = (
                session_context.session_config.session.delete_room_on_terminate
                if session_context.session_config
                else True
            )
            deleted = False
            if should_delete and hasattr(ctx, "delete_room"):
                try:
                    res = ctx.delete_room()
                    if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
                        await res
                    deleted = True
                except Exception as del_err:
                    logger.warning(f"Error deleting room on teardown: {del_err}")

            if not deleted and hasattr(ctx, "room") and ctx.room and hasattr(ctx.room, "disconnect"):
                try:
                    res = ctx.room.disconnect()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as disc_err:
                    logger.warning(f"Error disconnecting room on teardown: {disc_err}")
        except Exception as ex:
            logger.warning(f"Error disconnecting/deleting room: {ex}")

        # 7. Detach SessionContext
        detach_from_job(ctx)
        logger.info(f"Session teardown complete for job '{session_context.job_id}'")

    async def on_session_end(self, ctx: JobContext) -> None:
        """Callback invoked by LiveKit server when the RTC session concludes.

        Args:
            ctx: LiveKit JobContext for the concluding session.
        """
        await self.teardown(ctx)
