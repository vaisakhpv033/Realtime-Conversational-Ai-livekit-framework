"""Configurable EndCallTool with utterance-drain synchronization and room teardown."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
import logging
from typing import Any, Optional

from livekit.agents.job import JobContext, get_job_context
from livekit.agents.llm import RealtimeModel, ToolFlag, Toolset, function_tool
from livekit.agents.voice.events import CloseEvent, RunContext, SpeechCreatedEvent
from livekit.agents.voice.speech_handle import SpeechHandle

logger = logging.getLogger("agent_framework.tools.builtin.end_call")

DEFAULT_END_CALL_DESCRIPTION = (
    "Gracefully concludes the call and terminates the session when the conversation "
    "has finished or the user requests to hang up."
)

DEFAULT_FAREWELL_INSTRUCTION = "Say goodbye to the user briefly before hanging up."


def _safe_get_job_context() -> Optional[JobContext]:
    """Retrieves the active LiveKit JobContext if available, returning None otherwise."""
    try:
        return get_job_context()
    except RuntimeError:
        return None


class EndCallTool(Toolset):
    """Configurable tool allowing the voice agent to gracefully terminate a session.

    Features:
    - Configurable tool name (default 'end_call'), description, and farewell instructions.
    - Configurable room deletion policy (delete_room=True by default).
    - Utterance-drain synchronization: Ensures any in-flight or model-generated farewell
      speech completely plays out over WebRTC before the room is disconnected or deleted.
    - Graceful room disconnect and LiveKit room deletion via Server Room APIs.
    - Safe execution in both live worker environments and mock/unit test fixtures.
    """

    def __init__(
        self,
        *,
        name: str = "end_call",
        description: Optional[str] = None,
        farewell_instruction: Optional[str] = None,
        delete_room: bool = True,
        drain_timeout_seconds: float = 5.0,
        job_ctx: Optional[JobContext] = None,
        extra_description: str = "",
        ignore_on_enter: bool = False,
        end_instructions: Optional[str] = None,  # Backward-compatible alias
        on_tool_called: Optional[Callable[[Toolset.ToolCalledEvent], Awaitable[None] | None]] = None,
        on_tool_completed: Optional[Callable[[Toolset.ToolCompletedEvent], Awaitable[None] | None]] = None,
    ) -> None:
        """Initializes the EndCallTool.

        Args:
            name: Tool name registered for the LLM (default: 'end_call').
            description: Description provided to the LLM.
            farewell_instruction: Output directive instructing the model how to conclude.
            delete_room: Whether to delete the LiveKit room on disconnect.
            drain_timeout_seconds: Maximum seconds to wait for farewell audio playout.
            job_ctx: Optional explicit JobContext reference.
            extra_description: Supplementary description appended to the base description.
            ignore_on_enter: Suppresses tool availability during initial agent greeting.
            end_instructions: Alias for farewell_instruction for backward compatibility.
            on_tool_called: Optional async/sync callback triggered on invocation.
            on_tool_completed: Optional async/sync callback triggered upon completion.
        """
        self._name = name.strip() or "end_call"
        base_desc = description or DEFAULT_END_CALL_DESCRIPTION
        self._description = f"{base_desc}\n{extra_description}".strip() if extra_description else base_desc

        # Resolve farewell instruction
        if farewell_instruction is not None:
            self._farewell_instruction = farewell_instruction
        elif end_instructions is not None:
            self._farewell_instruction = end_instructions
        else:
            self._farewell_instruction = DEFAULT_FAREWELL_INSTRUCTION

        self._delete_room = delete_room
        self._drain_timeout_seconds = max(0.01, float(drain_timeout_seconds))
        self._job_ctx = job_ctx
        self._on_tool_called = on_tool_called
        self._on_tool_completed = on_tool_completed

        self._shutdown_session_task: Optional[asyncio.Task[None]] = None
        self._teardown_started = False
        self._room_deleted = False
        self._end_call_started = False

        flags = ToolFlag.IGNORE_ON_ENTER if ignore_on_enter else ToolFlag.NONE

        fn_tool = function_tool(
            self._end_call,
            name=self._name,
            description=self._description,
            flags=flags,
        )

        super().__init__(id=self._name, tools=[fn_tool])
        self._fn_tool = self.tools[0]

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def farewell_instruction(self) -> str:
        return self._farewell_instruction

    @property
    def end_instructions(self) -> str:
        """Backward-compatible alias for farewell_instruction."""
        return self._farewell_instruction

    @property
    def delete_room(self) -> bool:
        return self._delete_room

    @property
    def drain_timeout_seconds(self) -> float:
        return self._drain_timeout_seconds

    @property
    def job_ctx(self) -> Optional[JobContext]:
        return self._job_ctx

    async def execute(self, ctx: Any = None, **kwargs: Any) -> Any:
        """Executes the end call sequence directly."""
        return await self._end_call(context=ctx)

    async def __call__(self, ctx: Any = None, **kwargs: Any) -> Any:
        """Permits direct callable invocation in test fixtures and scripts."""
        return await self._end_call(context=ctx)

    async def _end_call(self, context: RunContext = None) -> Any:
        """Internal handler called when the LLM invokes the end_call function tool."""
        ctx = context
        logger.info(f"'{self._name}' tool called: initiating graceful utterance drain and room teardown")

        # Idempotency guard: prevent duplicate shutdown tasks on concurrent calls in the same turn
        if self._end_call_started:
            logger.info(f"'{self._name}' already initiated, returning farewell instruction")
            return self._farewell_instruction
        self._end_call_started = True

        job_ctx = self._job_ctx or _safe_get_job_context()
        if job_ctx is None and hasattr(ctx, "job_ctx"):
            job_ctx = ctx.job_ctx

        # Trigger on_tool_called callback if registered
        if self._on_tool_called:
            arguments_dict: dict[str, Any] = {}
            fnc = getattr(ctx, "function_call", None)
            if fnc is not None and hasattr(fnc, "arguments"):
                try:
                    if isinstance(fnc.arguments, str):
                        arguments_dict = json.loads(fnc.arguments) if fnc.arguments else {}
                    elif isinstance(fnc.arguments, dict):
                        arguments_dict = fnc.arguments
                except Exception:
                    pass
            res = self._on_tool_called(Toolset.ToolCalledEvent(ctx=ctx, arguments=arguments_dict))
            if asyncio.iscoroutine(res):
                await res

        # Extract session and speech handle from context
        session = getattr(ctx, "session", None)
        speech_handle = getattr(ctx, "speech_handle", None)
        if speech_handle is None and session is not None:
            speech_handle = getattr(session, "current_speech", None)

        # Inspect LLM capabilities to check for auto_tool_reply_generation
        auto_tool_reply = False
        try:
            if session is not None and hasattr(session, "current_agent") and session.current_agent:
                activity = getattr(session.current_agent, "_get_activity_or_raise", lambda: None)()
                if activity:
                    rt_session = getattr(activity, "realtime_llm_session", None)
                    llm_v = getattr(activity, "llm", None)
                    if rt_session and hasattr(rt_session, "capabilities"):
                        auto_tool_reply = getattr(rt_session.capabilities, "auto_tool_reply_generation", False)
                    elif llm_v and hasattr(llm_v, "capabilities"):
                        auto_tool_reply = getattr(llm_v.capabilities, "auto_tool_reply_generation", False)
        except Exception as e:
            logger.debug(f"Could not determine auto_tool_reply_generation: {e}")

        # Utterance drain synchronization
        if speech_handle is not None and not speech_handle.done():
            # Current speech utterance is in progress
            fallback_timer: Optional[asyncio.Task[None]] = None

            def _on_speech_done(_: Any = None) -> None:
                if fallback_timer is not None and not fallback_timer.done():
                    fallback_timer.cancel()
                if not auto_tool_reply:
                    self._shutdown(ctx, job_ctx)
                else:
                    self._shutdown_session_task = asyncio.create_task(
                        self._delayed_session_shutdown(ctx, job_ctx)
                    )

            if hasattr(speech_handle, "add_done_callback"):
                speech_handle.add_done_callback(_on_speech_done)

                # Timeout fallback: force shutdown if speech_handle hangs or emits silence
                async def _drain_timeout_watchdog() -> None:
                    try:
                        await asyncio.sleep(self._drain_timeout_seconds)
                        logger.warning(
                            f"Speech handle drain timed out after {self._drain_timeout_seconds}s; forcing shutdown."
                        )
                        _on_speech_done()
                    except asyncio.CancelledError:
                        pass

                fallback_timer = asyncio.create_task(_drain_timeout_watchdog())
                self._shutdown_session_task = fallback_timer
            else:
                self._shutdown_session_task = asyncio.create_task(
                    self._wait_speech_handle(speech_handle, ctx, job_ctx, auto_tool_reply)
                )
        elif auto_tool_reply and session is not None:
            # Model will generate farewell reply speech after this tool completes.
            # Register speech_created listener immediately to avoid race conditions.
            speech_created_fut: asyncio.Future[Any] = asyncio.Future()

            def _on_speech_created(ev: Any) -> None:
                if not speech_created_fut.done():
                    handle = getattr(ev, "speech_handle", ev)
                    speech_created_fut.set_result(handle)

            if hasattr(session, "once"):
                session.once("speech_created", _on_speech_created)

            self._shutdown_session_task = asyncio.create_task(
                self._wait_speech_created_and_shutdown(
                    speech_created_fut, session, _on_speech_created, ctx, job_ctx
                )
            )
        else:
            # No active speech handle or speech is already finished; drain audio buffer
            self._shutdown_session_task = asyncio.create_task(
                self._drain_and_shutdown(ctx, job_ctx)
            )

        # Listen for session close event
        if session is not None and hasattr(session, "once"):
            try:
                session.once("close", lambda ev: self._on_session_close(ev, job_ctx, ctx))
            except Exception as e:
                logger.debug(f"Could not attach session close listener: {e}")

        completed_ev = Toolset.ToolCompletedEvent(ctx=ctx, output=self._farewell_instruction)
        if self._on_tool_completed:
            res = self._on_tool_completed(completed_ev)
            if asyncio.iscoroutine(res):
                await res

        return completed_ev.output

    async def _wait_speech_created_and_shutdown(
        self,
        speech_created_fut: asyncio.Future[Any],
        session: Any,
        listener_cb: Callable[[Any], None],
        ctx: Any,
        job_ctx: Optional[JobContext],
    ) -> None:
        """Waits for the LLM farewell tool-reply speech to be created and play out before shutdown."""
        try:
            speech_handle = await asyncio.wait_for(speech_created_fut, timeout=self._drain_timeout_seconds)
            if hasattr(speech_handle, "wait_for_playout"):
                await asyncio.wait_for(speech_handle.wait_for_playout(), timeout=self._drain_timeout_seconds)
            elif hasattr(speech_handle, "__await__"):
                await asyncio.wait_for(speech_handle, timeout=self._drain_timeout_seconds)
            elif hasattr(speech_handle, "done") and not speech_handle.done():
                for _ in range(int(self._drain_timeout_seconds * 20)):
                    if speech_handle.done():
                        break
                    await asyncio.sleep(0.05)
        except asyncio.TimeoutError:
            logger.warning("Farewell utterance playout timed out, proceeding to shutdown")
        except Exception as e:
            logger.warning(f"Error awaiting farewell speech playout: {e}")
        finally:
            if session is not None and hasattr(session, "off"):
                try:
                    session.off("speech_created", listener_cb)
                except Exception:
                    pass
            self._shutdown(ctx, job_ctx)

    async def _wait_speech_handle(
        self,
        speech_handle: Any,
        ctx: Any,
        job_ctx: Optional[JobContext],
        auto_tool_reply: bool,
    ) -> None:
        """Waits for an active speech handle to complete before proceeding."""
        try:
            if hasattr(speech_handle, "wait_for_playout"):
                await asyncio.wait_for(speech_handle.wait_for_playout(), timeout=self._drain_timeout_seconds)
            elif hasattr(speech_handle, "__await__"):
                await asyncio.wait_for(speech_handle, timeout=self._drain_timeout_seconds)
            elif hasattr(speech_handle, "add_done_callback"):
                done_event = asyncio.Event()
                if hasattr(speech_handle, "done") and speech_handle.done():
                    done_event.set()
                else:
                    speech_handle.add_done_callback(lambda _: done_event.set())
                await asyncio.wait_for(done_event.wait(), timeout=self._drain_timeout_seconds)
            elif hasattr(speech_handle, "done"):
                for _ in range(int(self._drain_timeout_seconds * 20)):
                    if speech_handle.done():
                        break
                    await asyncio.sleep(0.05)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Timeout or error waiting for speech handle: {e}")

        session = getattr(ctx, "session", None)
        if auto_tool_reply and session is not None:
            speech_created_fut: asyncio.Future[Any] = asyncio.Future()

            def _on_speech_created(ev: Any) -> None:
                if not speech_created_fut.done():
                    handle = getattr(ev, "speech_handle", ev)
                    speech_created_fut.set_result(handle)

            if hasattr(session, "once"):
                session.once("speech_created", _on_speech_created)

            await self._wait_speech_created_and_shutdown(
                speech_created_fut, session, _on_speech_created, ctx, job_ctx
            )
        else:
            self._shutdown(ctx, job_ctx)

    async def _drain_and_shutdown(self, ctx: Any, job_ctx: Optional[JobContext]) -> None:
        """Drains the AgentSession audio buffer before shutting down."""
        session = getattr(ctx, "session", None)
        if session is not None and hasattr(session, "drain"):
            try:
                await asyncio.wait_for(session.drain(), timeout=self._drain_timeout_seconds)
            except Exception as e:
                logger.debug(f"Audio buffer drain completed with: {e}")

        self._shutdown(ctx, job_ctx)

    def _shutdown(self, ctx: Any, job_ctx: Optional[JobContext]) -> None:
        """Triggers session shutdown and schedules WebRTC room disconnect and deletion."""
        session = getattr(ctx, "session", None)
        if session is not None and hasattr(session, "shutdown"):
            try:
                session.shutdown()
            except Exception as e:
                logger.debug(f"Error shutting down session: {e}")

        # Execute room teardown asynchronously
        asyncio.create_task(self._disconnect_and_delete(ctx, job_ctx))

    async def _disconnect_and_delete(self, ctx: Any, job_ctx: Optional[JobContext]) -> None:
        """Disconnects the WebRTC room and deletes the room if delete_room=True."""
        if self._teardown_started:
            return
        self._teardown_started = True

        try:
            # 1. Locate room object
            room = None
            if hasattr(ctx, "room") and ctx.room is not None:
                room = ctx.room
            elif hasattr(ctx, "session") and hasattr(ctx.session, "room") and ctx.session.room is not None:
                room = ctx.session.room
            elif job_ctx is not None and hasattr(job_ctx, "room") and job_ctx.room is not None:
                room = job_ctx.room

            # 2. Disconnect local participant from room
            if room is not None and hasattr(room, "disconnect"):
                try:
                    res = room.disconnect()
                    if asyncio.iscoroutine(res):
                        await res
                    logger.info("Room participant disconnected successfully")
                except Exception as e:
                    logger.warning(f"Error disconnecting room: {e}")

            # 3. Delete room via LiveKit Room Service API if configured
            if self._delete_room and not self._room_deleted and job_ctx is not None and hasattr(job_ctx, "delete_room"):
                self._room_deleted = True
                try:
                    res = job_ctx.delete_room()
                    if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
                        await res
                    logger.info("LiveKit room deleted successfully")
                except Exception as e:
                    logger.warning(f"Error deleting LiveKit room: {e}")

            # 4. Shut down job process
            if job_ctx is not None and hasattr(job_ctx, "shutdown"):
                try:
                    job_ctx.shutdown(reason="end_call")
                except Exception as e:
                    logger.debug(f"Error shutting down job_ctx: {e}")

        except Exception as e:
            logger.warning(f"Error during disconnect_and_delete teardown: {e}")

    def _on_session_close(self, ev: CloseEvent, job_ctx: Optional[JobContext], ctx: Any) -> None:
        """Handles CloseEvent emitted by AgentSession."""
        if self._shutdown_session_task:
            self._shutdown_session_task.cancel()
            self._shutdown_session_task = None

        if self._delete_room and not self._room_deleted and job_ctx is not None:
            self._room_deleted = True
            async def _on_shutdown() -> None:
                logger.info("Deleting room on job shutdown because end_call was invoked")
                try:
                    res = job_ctx.delete_room()
                    if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
                        await res
                except Exception as e:
                    logger.warning(f"Error in delete_room shutdown callback: {e}")

            if hasattr(job_ctx, "add_shutdown_callback"):
                job_ctx.add_shutdown_callback(_on_shutdown)

        if job_ctx is not None and hasattr(job_ctx, "shutdown"):
            reason = getattr(ev, "reason", None)
            reason_str = getattr(reason, "value", str(reason)) if reason is not None else "end_call"
            try:
                job_ctx.shutdown(reason=reason_str)
            except Exception as e:
                logger.debug(f"Error shutting down job_ctx on session close: {e}")

