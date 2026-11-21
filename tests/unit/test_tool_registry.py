"""Unit tests for Phase 4: Dynamic Tool Registry & EndCall Generalization.

Covers:
1. EndCallTool creation with custom farewell instructions, delete_room flags, and aliases.
2. EndCallTool execution, audio drain synchronization, delayed reply handling, and room teardown.
3. WebhookTool creation from JSON schema parameters and ToolConfig models.
4. WebhookTool async execution with local HTTP server (GET/POST/PUT, query params, body, secrets in headers).
5. SSRF validation rejection of cloud metadata and private IP endpoints.
6. Graceful error handling in WebhookTool (HTTP 500/404, timeouts, non-JSON responses).
7. ToolRegistry.build_tools instantiating declared tools from SessionConfig.
8. Registration and lifecycle of custom builtin tools in ToolRegistry.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import aiohttp
from aiohttp import web
import pytest

from agent_framework.config.schema import (
    AgentConfig,
    InstructionsConfig,
    ModelsConfig,
    ReportingConfig,
    ReportingEndpointConfig,
    ReportingSecurityConfig,
    SessionConfig,
    ToolConfig,
    VoiceConfig,
)
from agent_framework.tools import (
    DEFAULT_END_CALL_DESCRIPTION,
    DEFAULT_FAREWELL_INSTRUCTION,
    EndCallTool,
    SSRFSecurityError,
    ToolRegistry,
    WebhookConnectionError,
    WebhookHTTPError,
    WebhookTimeoutError,
    WebhookTool,
    create_tool,
    default_tool_registry,
)
from livekit.agents.llm import FunctionToolCall, ToolContext, execute_function_call
from livekit.agents.voice.events import RunContext


# ==============================================================================
# Ephemeral Local HTTP Test Server Context Manager
# ==============================================================================
class LocalHttpServer:
    """Ephemeral in-process aiohttp test server for webhook tests."""

    def __init__(self) -> None:
        self.captured_requests: list[dict[str, Any]] = []
        self.url: str = ""
        self.requests = self.captured_requests
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    async def __aenter__(self) -> LocalHttpServer:
        app = web.Application()

        async def handle_get(request: web.Request) -> web.Response:
            self.captured_requests.append({
                "method": request.method,
                "path": request.path,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "query": dict(request.query),
                "query_all": {k: request.query.getall(k) for k in request.query},
            })
            return web.json_response({
                "status": "ok",
                "query": dict(request.query),
                "query_all": {k: request.query.getall(k) for k in request.query},
            })

        async def handle_post_put(request: web.Request) -> web.Response:
            data = await request.json() if request.can_read_body else {}
            self.captured_requests.append({
                "method": request.method,
                "path": request.path,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "body": data,
            })
            return web.json_response({"status": "received", "data": data, "method": request.method})

        async def handle_user_path(request: web.Request) -> web.Response:
            user_id = request.match_info.get("user_id", "")
            data = await request.json() if request.can_read_body and request.content_length else {}
            self.captured_requests.append({
                "method": request.method,
                "path": request.path,
                "raw_path": request.raw_path,
                "user_id": user_id,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "body": data,
            })
            return web.json_response({"user_id": user_id, "found": True})

        async def handle_error_500(request: web.Request) -> web.Response:
            return web.Response(text="Internal Server Error: Database failure", status=500)

        async def handle_slow_timeout(request: web.Request) -> web.Response:
            await asyncio.sleep(0.5)
            return web.json_response({"status": "too_late"})

        async def handle_raw_text(request: web.Request) -> web.Response:
            return web.Response(text="Service is healthy (plain text)", status=200, content_type="text/plain")

        async def handle_large_payload(request: web.Request) -> web.Response:
            large_text = "ABCDEFGHIJ" * 2000  # 20,000 chars
            return web.Response(text=large_text, status=200, content_type="text/plain")

        app.router.add_get("/api/items", handle_get)
        app.router.add_post("/api/items", handle_post_put)
        app.router.add_put("/api/items", handle_post_put)
        app.router.add_post("/api/users/{user_id}/records", handle_user_path)
        app.router.add_get("/api/error500", handle_error_500)
        app.router.add_get("/api/slow", handle_slow_timeout)
        app.router.add_get("/api/text", handle_raw_text)
        app.router.add_get("/api/large", handle_large_payload)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()

        # Retrieve dynamic OS assigned port
        port = self._site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


@pytest.fixture
def base_session_config() -> SessionConfig:
    """Provides a valid baseline SessionConfig."""
    return SessionConfig(
        agent=AgentConfig(name="test-agent", role="receptionist"),
        models=ModelsConfig(provider="gemini", model="gemini-3.8-live"),
        instructions=InstructionsConfig(system_prompt="You are a helpful assistant."),
        voice=VoiceConfig(provider="gemini", voice_id="Aoede"),
        reporting=ReportingConfig(
            enabled=False,
            security=ReportingSecurityConfig(allow_local=True),
        ),
    )


# ==============================================================================
# 1. EndCallTool Unit Tests
# ==============================================================================
class TestEndCallTool:
    """Verifies EndCallTool configuration, execution, audio drain, and room teardown."""

    def test_end_call_tool_defaults(self):
        """Verifies default properties of EndCallTool."""
        tool = EndCallTool()
        assert tool.name == "end_call"
        assert tool.delete_room is True
        assert tool.drain_timeout_seconds == 5.0
        assert tool.farewell_instruction == DEFAULT_FAREWELL_INSTRUCTION
        assert tool.description == DEFAULT_END_CALL_DESCRIPTION
        assert len(tool.tools) == 1
        assert tool.tools[0].info.name == "end_call"

    def test_end_call_tool_custom_configuration(self):
        """Verifies custom name, description, farewell instruction, and delete_room flags."""
        tool = EndCallTool(
            name="terminate_call",
            description="Closes the patient call.",
            farewell_instruction="Say 'Thank you for choosing our clinic. Have a wonderful day!'",
            delete_room=False,
            drain_timeout_seconds=2.5,
            extra_description="Call only when user indicates they wish to finish.",
        )
        assert tool.name == "terminate_call"
        assert tool.delete_room is False
        assert tool.drain_timeout_seconds == 2.5
        assert tool.farewell_instruction == "Say 'Thank you for choosing our clinic. Have a wonderful day!'"
        assert "Closes the patient call." in tool.description
        assert "Call only when user indicates they wish to finish." in tool.description
        assert len(tool.tools) == 1
        assert tool.tools[0].info.name == "terminate_call"

    def test_end_call_tool_alias_end_instructions(self):
        """Verifies backward compatibility with end_instructions argument."""
        tool = EndCallTool(end_instructions="Say goodbye kindly.")
        assert tool.farewell_instruction == "Say goodbye kindly."
        assert tool.end_instructions == "Say goodbye kindly."

    @pytest.mark.asyncio
    async def test_end_call_execution_standalone(self):
        """Verifies direct execution when no active speech or session is present."""
        tool = EndCallTool(farewell_instruction="Bye now!")
        mock_ctx = MagicMock()
        mock_ctx.session = None
        mock_ctx.speech_handle = None

        result = await tool(mock_ctx)
        assert result == "Bye now!"

    @pytest.mark.asyncio
    async def test_end_call_execution_with_speech_handle_drain(self):
        """Verifies utterance drain callback is attached when speech handle is active."""
        tool = EndCallTool(delete_room=True)

        mock_speech_handle = MagicMock()
        mock_speech_handle.done.return_value = False
        callbacks: list[Any] = []
        mock_speech_handle.add_done_callback.side_effect = lambda cb: callbacks.append(cb)

        mock_room = MagicMock()
        mock_room.disconnect = AsyncMock()

        mock_job_ctx = MagicMock()
        mock_job_ctx.room = mock_room
        mock_job_ctx.delete_room = AsyncMock()
        mock_job_ctx.shutdown = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.speech_handle = mock_speech_handle
        mock_ctx.session = None
        mock_ctx.room = mock_room
        mock_ctx.job_ctx = mock_job_ctx

        # Execute tool
        result = await tool(mock_ctx)
        assert result == DEFAULT_FAREWELL_INSTRUCTION
        assert len(callbacks) == 1

        # Simulate completion of the farewell speech utterance
        callbacks[0](mock_speech_handle)

        # Allow scheduled teardown coroutine to execute
        await asyncio.sleep(0.05)

        mock_room.disconnect.assert_awaited_once()
        mock_job_ctx.delete_room.assert_awaited_once()
        mock_job_ctx.shutdown.assert_called_once_with(reason="end_call")

    @pytest.mark.asyncio
    async def test_end_call_drain_timeout_fallback_when_speech_handle_hangs(self):
        """Verifies drain_timeout_seconds forces teardown if speech_handle never completes."""
        tool = EndCallTool(delete_room=True, drain_timeout_seconds=0.1)

        mock_speech_handle = MagicMock()
        mock_speech_handle.done.return_value = False
        callbacks: list[Any] = []
        mock_speech_handle.add_done_callback.side_effect = lambda cb: callbacks.append(cb)

        mock_room = MagicMock()
        mock_room.disconnect = AsyncMock()

        mock_job_ctx = MagicMock()
        mock_job_ctx.room = mock_room
        mock_job_ctx.delete_room = AsyncMock()
        mock_job_ctx.shutdown = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.speech_handle = mock_speech_handle
        mock_ctx.session = None
        mock_ctx.room = mock_room
        mock_ctx.job_ctx = mock_job_ctx

        # Execute tool - callback attached, but we deliberately DO NOT invoke callback
        result = await tool(mock_ctx)
        assert result == DEFAULT_FAREWELL_INSTRUCTION
        assert len(callbacks) == 1

        # Wait for drain_timeout_seconds (0.1s) to elapse
        await asyncio.sleep(0.18)

        # Forced shutdown should have completed
        mock_room.disconnect.assert_awaited_once()
        mock_job_ctx.delete_room.assert_awaited_once()
        mock_job_ctx.shutdown.assert_called_once_with(reason="end_call")

    @pytest.mark.asyncio
    async def test_end_call_delayed_reply_shutdown(self):
        """Verifies delayed session shutdown waits for auto_tool_reply_generation speech."""
        tool = EndCallTool(delete_room=True, drain_timeout_seconds=2.0)

        mock_room = MagicMock()
        mock_room.disconnect = AsyncMock()

        mock_job_ctx = MagicMock()
        mock_job_ctx.room = mock_room
        mock_job_ctx.delete_room = AsyncMock()
        mock_job_ctx.shutdown = MagicMock()

        # Session with auto_tool_reply_generation enabled
        mock_activity = MagicMock()
        mock_activity.realtime_llm_session.capabilities.auto_tool_reply_generation = True

        mock_session = MagicMock()
        mock_session.current_agent._get_activity_or_raise.return_value = mock_activity
        mock_session.shutdown = MagicMock()

        speech_created_callbacks: list[Any] = []

        def mock_once(event_name: str, cb: Any):
            if event_name == "speech_created":
                speech_created_callbacks.append(cb)

        mock_session.once.side_effect = mock_once

        mock_ctx = MagicMock()
        mock_ctx.speech_handle = None
        mock_ctx.session = mock_session
        mock_ctx.room = mock_room
        mock_ctx.job_ctx = mock_job_ctx

        # Execute tool
        await tool(mock_ctx)
        assert len(speech_created_callbacks) == 1

        # Simulate speech_created event for the model's goodbye message
        mock_farewell_speech = MagicMock()
        mock_farewell_speech.wait_for_playout = AsyncMock()
        mock_farewell_event = MagicMock()
        mock_farewell_event.speech_handle = mock_farewell_speech

        speech_created_callbacks[0](mock_farewell_event)

        # Allow delayed shutdown task to process playout
        await asyncio.sleep(0.05)

        mock_farewell_speech.wait_for_playout.assert_awaited_once()
        mock_session.shutdown.assert_called_once()
        mock_room.disconnect.assert_awaited_once()
        mock_job_ctx.delete_room.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_end_call_delete_room_disabled(self):
        """Verifies delete_room=False disconnects room but skips room deletion."""
        tool = EndCallTool(delete_room=False)

        mock_room = MagicMock()
        mock_room.disconnect = AsyncMock()

        mock_job_ctx = MagicMock()
        mock_job_ctx.room = mock_room
        mock_job_ctx.delete_room = AsyncMock()
        mock_job_ctx.shutdown = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.speech_handle = None
        mock_ctx.session = None
        mock_ctx.room = mock_room
        mock_ctx.job_ctx = mock_job_ctx

        await tool(mock_ctx)
        await asyncio.sleep(0.05)

        mock_room.disconnect.assert_awaited_once()
        mock_job_ctx.delete_room.assert_not_called()
        mock_job_ctx.shutdown.assert_called_once_with(reason="end_call")

    @pytest.mark.asyncio
    async def test_end_call_callbacks_invoked(self):
        """Verifies on_tool_called and on_tool_completed callbacks."""
        called_args = []
        completed_args = []

        async def on_called(ev):
            called_args.append(ev)

        async def on_completed(ev):
            completed_args.append(ev)

        tool = EndCallTool(
            on_tool_called=on_called,
            on_tool_completed=on_completed,
            farewell_instruction="Finished",
        )
        mock_ctx = MagicMock()
        mock_ctx.session = None
        mock_ctx.speech_handle = None

        result = await tool(mock_ctx)
        assert result == "Finished"
        assert len(called_args) == 1
        assert len(completed_args) == 1
        assert completed_args[0].output == "Finished"

    @pytest.mark.asyncio
    async def test_end_call_livekit_function_call_integration(self):
        """Verifies EndCallTool executes cleanly via LiveKit execute_function_call with RunContext."""
        tool = EndCallTool(farewell_instruction="Goodbye! Call back anytime.")
        tool_ctx = ToolContext([tool])
        call = FunctionToolCall(call_id="call_end_001", name="end_call", arguments="{}")

        mock_speech = MagicMock()
        mock_speech.num_steps = 1
        mock_run_ctx = RunContext(session=MagicMock(), speech_handle=mock_speech, function_call=MagicMock())
        mock_run_ctx._updates = []

        result = await execute_function_call(call, tool_ctx, call_ctx=mock_run_ctx)
        assert result.fnc_call_out.is_error is False
        assert result.fnc_call_out.output == "Goodbye! Call back anytime."

    @pytest.mark.asyncio
    async def test_end_call_livekit_function_call_with_extra_arguments(self):
        """Verifies EndCallTool handles arbitrary LLM-generated arguments gracefully."""
        tool = EndCallTool(farewell_instruction="Farewell.")
        tool_ctx = ToolContext([tool])
        call = FunctionToolCall(
            call_id="call_end_002",
            name="end_call",
            arguments=json.dumps({"reason": "user_hung_up", "timestamp": "2026-11-20T12:00:00Z"}),
        )

        result = await execute_function_call(call, tool_ctx, call_ctx=None)
        assert result.fnc_call_out.is_error is False
        assert result.fnc_call_out.output == "Farewell."

    @pytest.mark.asyncio
    async def test_end_call_concurrent_invocation_idempotency(self):
        """Verifies duplicate concurrent invocations of end_call only trigger teardown once."""
        tool = EndCallTool(delete_room=True)
        mock_room = MagicMock()
        mock_room.disconnect = AsyncMock()

        mock_job_ctx = MagicMock()
        mock_job_ctx.room = mock_room
        mock_job_ctx.delete_room = AsyncMock()
        mock_job_ctx.shutdown = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.session = None
        mock_ctx.speech_handle = None
        mock_ctx.room = mock_room
        mock_ctx.job_ctx = mock_job_ctx

        # Execute duplicate calls concurrently
        res1, res2 = await asyncio.gather(tool(mock_ctx), tool(mock_ctx))
        assert res1 == DEFAULT_FAREWELL_INSTRUCTION
        assert res2 == DEFAULT_FAREWELL_INSTRUCTION

        await asyncio.sleep(0.05)

        # Disconnect and delete should only be awaited once
        assert mock_room.disconnect.await_count == 1
        assert mock_job_ctx.delete_room.await_count == 1


# ==============================================================================
# 2. WebhookTool Unit Tests
# ==============================================================================
class TestWebhookTool:
    """Verifies WebhookTool parameter schemas, HTTP execution, secrets, and SSRF security."""

    def test_webhook_tool_init_from_parameters(self):
        """Verifies WebhookTool creates a valid RawFunctionTool matching parameter schema."""
        params = {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient identifier"},
                "appointment_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["patient_id"],
        }
        tool = WebhookTool(
            name="book_appointment",
            description="Books a dental appointment for a patient.",
            parameters=params,
            url="https://api.clinic.com/v1/appointments",
            method="POST",
            headers={"X-Version": "2.0"},
            timeout_seconds=7.5,
        )

        assert tool.name == "book_appointment"
        assert tool.description == "Books a dental appointment for a patient."
        assert tool.method == "POST"
        assert tool.timeout_seconds == 7.5
        assert tool.headers == {"X-Version": "2.0"}
        assert len(tool.tools) == 1
        assert tool.function_tool.info.name == "book_appointment"
        assert tool.function_tool.info.raw_schema["parameters"] == params

    def test_webhook_tool_init_from_tool_config(self):
        """Verifies WebhookTool construction from a validated ToolConfig schema."""
        tool_cfg = ToolConfig(
            name="get_patient_records",
            type="webhook",
            description="Retrieves records by ID.",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
            config={
                "url": "https://api.clinic.com/v1/records",
                "method": "GET",
                "headers": {"Authorization": "Bearer {{env:CLINIC_TOKEN}}"},
                "timeout_seconds": 12.0,
            },
        )
        tool = WebhookTool.from_config(tool_cfg)
        assert tool.name == "get_patient_records"
        assert tool.method == "GET"
        assert tool.url == "https://api.clinic.com/v1/records"
        assert tool.timeout_seconds == 12.0
        assert tool.headers["Authorization"] == "Bearer {{env:CLINIC_TOKEN}}"

    def test_webhook_tool_invalid_configurations_raise(self):
        """Verifies validation of empty names, missing URLs, or unsupported HTTP methods."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            WebhookTool(name="", url="https://api.example.com")

        with pytest.raises(ValueError, match="must specify a target URL"):
            WebhookTool(name="my_tool", url="")

        with pytest.raises(ValueError, match="Invalid HTTP method 'DELETE'"):
            WebhookTool(name="my_tool", url="https://api.example.com", method="DELETE")

    @pytest.mark.asyncio
    async def test_webhook_tool_async_get_with_query_params(self):
        """Verifies GET requests serialize arguments as URL query parameters."""
        async with LocalHttpServer() as local_http_server:
            tool = WebhookTool(
                name="search_items",
                url=f"{local_http_server.url}/api/items",
                method="GET",
                allow_local=True,
            )

            response_str = await tool(category="dental", in_stock=True)
            response_json = json.loads(response_str)

            assert response_json["status"] == "ok"
            assert response_json["query"]["category"] == "dental"
            assert response_json["query"]["in_stock"] == "true"

            # Verify server received GET request with query params
            assert len(local_http_server.requests) == 1
            req = local_http_server.requests[0]
            assert req["method"] == "GET"
            assert req["query"] == {"category": "dental", "in_stock": "true"}

    @pytest.mark.asyncio
    async def test_webhook_tool_async_post_with_json_body(self):
        """Verifies POST requests serialize arguments as JSON body."""
        async with LocalHttpServer() as local_http_server:
            tool = WebhookTool(
                name="create_item",
                url=f"{local_http_server.url}/api/items",
                method="POST",
                allow_local=True,
            )

            payload = {"item_name": "Composite Resin", "sku": "CR-101", "quantity": 5}
            response_str = await tool(payload)
            response_json = json.loads(response_str)

            assert response_json["status"] == "received"
            assert response_json["method"] == "POST"
            assert response_json["data"] == payload

            assert len(local_http_server.requests) == 1
            req = local_http_server.requests[0]
            assert req["method"] == "POST"
            assert req["body"] == payload

    @pytest.mark.asyncio
    async def test_webhook_tool_async_put_with_json_body(self):
        """Verifies PUT requests serialize arguments as JSON body."""
        async with LocalHttpServer() as local_http_server:
            tool = WebhookTool(
                name="update_item",
                url=f"{local_http_server.url}/api/items",
                method="PUT",
                allow_local=True,
            )

            update_data = {"sku": "CR-101", "quantity": 10}
            response_str = await tool(**update_data)
            response_json = json.loads(response_str)

            assert response_json["status"] == "received"
            assert response_json["method"] == "PUT"
            assert response_json["data"] == update_data

    @pytest.mark.asyncio
    async def test_webhook_tool_path_param_interpolation(self):
        """Verifies {param} in URL is substituted and removed from the request body."""
        async with LocalHttpServer() as local_http_server:
            tool = WebhookTool(
                name="add_user_record",
                url=f"{local_http_server.url}/api/users/{{user_id}}/records",
                method="POST",
                allow_local=True,
            )

            args = {"user_id": "USR-4081", "diagnosis": "Gingivitis", "notes": "Mild inflammation"}
            response_str = await tool(args)
            response_json = json.loads(response_str)

            assert response_json["user_id"] == "USR-4081"
            assert response_json["found"] is True

            assert len(local_http_server.requests) == 1
            req = local_http_server.requests[0]
            assert req["path"] == "/api/users/USR-4081/records"
            # user_id should be extracted into URL path, leaving remaining fields in body
            assert req["body"] == {"diagnosis": "Gingivitis", "notes": "Mild inflammation"}

    @pytest.mark.asyncio
    async def test_webhook_tool_custom_headers_with_secret_substitution(self):
        """Verifies {{env:KEY}} references in headers are resolved before dispatch."""
        async with LocalHttpServer() as local_http_server:
            mock_env = {
                "API_KEY": "secret_live_token_777",
                "TENANT_ID": "tenant_citrus_smiles",
            }
            tool = WebhookTool(
                name="authenticated_call",
                url=f"{local_http_server.url}/api/items",
                method="GET",
                headers={
                    "Authorization": "Bearer {{env:API_KEY}}",
                    "X-Tenant": "{{env:TENANT_ID}}",
                    "X-Static": "static-value",
                },
                env=mock_env,
                allow_local=True,
            )

            await tool(q="search_test")
            assert len(local_http_server.requests) == 1
            headers = local_http_server.requests[0]["headers"]

            assert headers["authorization"] == "Bearer secret_live_token_777"
            assert headers["x-tenant"] == "tenant_citrus_smiles"
            assert headers["x-static"] == "static-value"

    @pytest.mark.asyncio
    async def test_webhook_tool_ssrf_rejection_metadata_ip(self):
        """Verifies requests to cloud metadata (169.254.169.254) are rejected by SSRF filter."""
        tool = WebhookTool(
            name="malicious_metadata_tool",
            url="http://169.254.169.254/latest/meta-data/",
            method="GET",
            allow_local=False,
        )

        # Default execution returns error JSON without raising exception
        response_str = await tool()
        response_json = json.loads(response_str)

        assert response_json["status"] == "blocked"
        assert response_json["status_code"] == 400
        assert "SSRF security check rejected" in response_json["error"]

        # With raise_on_error=True, raises SSRFSecurityError
        tool_strict = WebhookTool(
            name="malicious_strict",
            url="http://169.254.169.254/latest/meta-data/",
            method="GET",
            raise_on_error=True,
        )
        with pytest.raises(SSRFSecurityError, match="SSRF security check rejected"):
            await tool_strict()

    @pytest.mark.asyncio
    async def test_webhook_tool_ssrf_rejection_private_ip_when_not_allowed(self):
        """Verifies RFC-1918 private IP addresses are rejected when allow_local=False."""
        tool = WebhookTool(
            name="private_network_tool",
            url="http://192.168.1.50:8080/internal/api",
            method="POST",
            allow_local=False,
        )

        response_str = await tool({"secret": "leak"})
        response_json = json.loads(response_str)

        assert response_json["status"] == "blocked"
        assert "Private network IP address" in response_json["error"]

    @pytest.mark.asyncio
    async def test_webhook_tool_http_500_graceful_handling(self):
        """Verifies HTTP 500 responses return structured error JSON without crashing."""
        async with LocalHttpServer() as local_http_server:
            tool = WebhookTool(
                name="failing_tool",
                url=f"{local_http_server.url}/api/error500",
                method="GET",
                allow_local=True,
            )

            response_str = await tool()
            response_json = json.loads(response_str)

            assert response_json["status_code"] == 500
            assert "returned HTTP 500" in response_json["error"]
            assert "Database failure" in response_json["response"]

            # Strict mode raises WebhookHTTPError
            tool_strict = WebhookTool(
                name="failing_strict",
                url=f"{local_http_server.url}/api/error500",
                method="GET",
                allow_local=True,
                raise_on_error=True,
            )
            with pytest.raises(WebhookHTTPError) as exc:
                await tool_strict()
            assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_webhook_tool_timeout_handling(self):
        """Verifies requests exceeding timeout_seconds return HTTP 504 error JSON."""
        async with LocalHttpServer() as local_http_server:
            tool = WebhookTool(
                name="slow_tool",
                url=f"{local_http_server.url}/api/slow",
                method="GET",
                timeout_seconds=0.1,  # server sleeps for 0.5s
                allow_local=True,
            )

            response_str = await tool()
            response_json = json.loads(response_str)

            assert response_json["status_code"] == 504
            assert "timed out after 0.1s" in response_json["error"]

            # Strict mode raises WebhookTimeoutError
            tool_strict = WebhookTool(
                name="slow_strict",
                url=f"{local_http_server.url}/api/slow",
                method="GET",
                timeout_seconds=0.1,
                allow_local=True,
                raise_on_error=True,
            )
            with pytest.raises(WebhookTimeoutError):
                await tool_strict()

    @pytest.mark.asyncio
    async def test_webhook_tool_invalid_json_response_handling(self):
        """Verifies plain text or invalid JSON responses are wrapped safely."""
        async with LocalHttpServer() as local_http_server:
            tool = WebhookTool(
                name="plain_text_tool",
                url=f"{local_http_server.url}/api/text",
                method="GET",
                allow_local=True,
            )

            response_str = await tool()
            response_json = json.loads(response_str)

            assert response_json["status_code"] == 200
            assert "Service is healthy (plain text)" in response_json["response"]

    @pytest.mark.asyncio
    async def test_webhook_tool_livekit_function_call_integration(self):
        """Verifies WebhookTool integrates seamlessly with LiveKit execute_function_call."""
        async with LocalHttpServer() as local_http_server:
            params = {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                },
                "required": ["category"],
            }
            tool = WebhookTool(
                name="lookup_items",
                description="Look up available items by category",
                parameters=params,
                url=f"{local_http_server.url}/api/items",
                method="GET",
                allow_local=True,
            )

            tool_ctx = ToolContext([tool])
            call = FunctionToolCall(
                call_id="call_abc_123",
                name="lookup_items",
                arguments=json.dumps({"category": "orthodontics"}),
            )

            result = await execute_function_call(call, tool_ctx)
            output_json = json.loads(result.fnc_call_out.output)

            assert output_json["status"] == "ok"
            assert output_json["query"]["category"] == "orthodontics"

    @pytest.mark.asyncio
    async def test_webhook_tool_path_param_url_encoding(self):
        """Verifies special characters and spaces in URL path parameters are safely URL-encoded."""
        async with LocalHttpServer() as local_http_server:
            tool = WebhookTool(
                name="user_lookup",
                url=f"{local_http_server.url}/api/users/{{user_id}}/records",
                method="POST",
                allow_local=True,
            )

            # user_id containing space and special characters
            res = await tool({"user_id": "Dr John Doe"})
            data = json.loads(res)
            assert data["found"] is True
            assert data["user_id"] == "Dr John Doe"
            assert local_http_server.requests[0]["raw_path"] == "/api/users/Dr%20John%20Doe/records"

    @pytest.mark.asyncio
    async def test_webhook_tool_missing_path_param_rejected(self):
        """Verifies unreplaced path parameters in URL return a structured 400 error."""
        tool = WebhookTool(
            name="missing_param_tool",
            url="https://api.example.com/users/{user_id}/records",
            method="GET",
            allow_local=True,
        )

        res = await tool({"unrelated_param": "abc"})
        data = json.loads(res)
        assert data["status_code"] == 400
        assert "Missing required URL path parameter" in data["error"]
        assert "user_id" in data["error"]

    @pytest.mark.asyncio
    async def test_webhook_tool_get_sequence_query_params(self):
        """Verifies sequence of primitive query parameters are serialized properly in GET requests."""
        async with LocalHttpServer() as local_http_server:
            tool = WebhookTool(
                name="tag_search",
                url=f"{local_http_server.url}/api/items",
                method="GET",
                allow_local=True,
            )

            res = await tool(tags=["urgent", "in-person"], limit=5)
            data = json.loads(res)
            assert data["status"] == "ok"
            assert local_http_server.requests[0]["query_all"]["tags"] == ["urgent", "in-person"]

    @pytest.mark.asyncio
    async def test_webhook_tool_response_truncation(self):
        """Verifies responses exceeding max_response_chars are cleanly truncated."""
        async with LocalHttpServer() as local_http_server:
            tool = WebhookTool(
                name="large_payload_tool",
                url=f"{local_http_server.url}/api/large",
                method="GET",
                allow_local=True,
                max_response_chars=500,
            )

            res = await tool()
            data = json.loads(res)
            assert data["status_code"] == 200
            assert "[truncated]" in data["response"]
            assert len(data["response"]) <= 600


# ==============================================================================
# 3. ToolRegistry Unit Tests
# ==============================================================================
class TestToolRegistry:
    """Verifies ToolRegistry tool building, custom builtin registration, and schema synchronization."""

    def test_registry_build_end_call_tool(self, base_session_config):
        """Verifies ToolRegistry builds EndCallTool from builtin declaration."""
        base_session_config.tools = [
            ToolConfig(
                name="end_call",
                type="builtin",
                description="Terminate call after goodbye.",
                config={
                    "delete_room": True,
                    "farewell_instruction": "Say 'Goodbye and take care!'",
                    "drain_timeout_seconds": 3.0,
                },
            )
        ]

        registry = ToolRegistry()
        tools = registry.build_tools(base_session_config)

        assert len(tools) == 1
        end_call_tool = tools[0]
        assert isinstance(end_call_tool, EndCallTool)
        assert end_call_tool.name == "end_call"
        assert end_call_tool.delete_room is True
        assert end_call_tool.drain_timeout_seconds == 3.0
        assert end_call_tool.farewell_instruction == "Say 'Goodbye and take care!'"

    def test_registry_build_webhook_tool(self, base_session_config):
        """Verifies ToolRegistry builds WebhookTool from webhook declaration."""
        base_session_config.tools = [
            ToolConfig(
                name="query_calendar",
                type="webhook",
                description="Query calendar availability",
                parameters={"type": "object", "properties": {"date": {"type": "string"}}},
                config={
                    "url": "https://calendar.clinic.com/api/slots",
                    "method": "GET",
                    "headers": {"X-Org": "CitrusSmiles"},
                    "timeout_seconds": 8.0,
                },
            )
        ]

        registry = ToolRegistry()
        tools = registry.build_tools(base_session_config)

        assert len(tools) == 1
        webhook_tool = tools[0]
        assert isinstance(webhook_tool, WebhookTool)
        assert webhook_tool.name == "query_calendar"
        assert webhook_tool.method == "GET"
        assert webhook_tool.url == "https://calendar.clinic.com/api/slots"
        assert webhook_tool.timeout_seconds == 8.0

    def test_registry_build_multiple_tools(self, base_session_config):
        """Verifies building both builtin and webhook tools concurrently."""
        base_session_config.tools = [
            ToolConfig(
                name="end_call",
                type="builtin",
            ),
            ToolConfig(
                name="lookup_patient",
                type="webhook",
                config={"url": "https://api.example.com/patients"},
            ),
            ToolConfig(
                name="send_sms",
                type="webhook",
                config={"url": "https://api.example.com/sms"},
            ),
        ]

        registry = ToolRegistry()
        tools = registry.build_tools(base_session_config)

        assert len(tools) == 3
        names = [t.name for t in tools]
        assert names == ["end_call", "lookup_patient", "send_sms"]

    def test_registry_duplicate_tool_name_rejected(self, base_session_config):
        """Verifies duplicate tool names raise ValueError during build_tools."""
        # Bypass schema validator to test defensive check in registry.build_tools
        t1 = ToolConfig(name="check_status", type="webhook", config={"url": "https://a.com"})
        t2 = ToolConfig(name="check_status", type="webhook", config={"url": "https://b.com"})
        base_session_config.tools = [t1]

        registry = ToolRegistry()
        # Mocking duplicate in session_config.tools
        base_session_config.__dict__["tools"] = [t1, t2]

        with pytest.raises(ValueError, match="Duplicate tool name 'check_status'"):
            registry.build_tools(base_session_config)

    def test_registry_custom_builtin_registration(self, base_session_config):
        """Verifies registering a custom builtin tool factory and schema synchronization."""
        def custom_transfer_factory(cfg: ToolConfig, ctx: Optional[Any] = None) -> Any:
            mock_tool = MagicMock()
            mock_tool.name = cfg.name
            mock_tool.target_dept = cfg.config.get("department", "billing")
            return mock_tool

        registry = ToolRegistry()
        registry.register_builtin("transfer_department", custom_transfer_factory)

        assert registry.is_builtin_registered("transfer_department")
        assert "transfer_department" in registry.get_registered_builtins()

        # Build session config with newly registered builtin
        base_session_config.tools = [
            ToolConfig(
                name="transfer_department",
                type="builtin",
                config={"department": "dental_hygiene"},
            )
        ]

        tools = registry.build_tools(base_session_config)
        assert len(tools) == 1
        assert tools[0].name == "transfer_department"
        assert tools[0].target_dept == "dental_hygiene"

        # Cleanup
        registry.unregister_builtin("transfer_department")
        assert not registry.is_builtin_registered("transfer_department")

    def test_registry_unknown_builtin_tool_raises(self):
        """Verifies create_tool raises ValueError when encountering an unregistered builtin tool."""
        registry = ToolRegistry()
        cfg = MagicMock()
        cfg.name = "unregistered_tool"
        cfg.type = "builtin"

        with pytest.raises(ValueError, match="Unknown builtin tool 'unregistered_tool'"):
            registry.create_tool(cfg)

    def test_registry_unknown_tool_type_raises(self):
        """Verifies create_tool raises ValueError when encountering an unknown tool type."""
        registry = ToolRegistry()
        cfg = MagicMock()
        cfg.name = "my_tool"
        cfg.type = "grpc_service"

        with pytest.raises(ValueError, match="Unsupported tool type 'grpc_service'"):
            registry.create_tool(cfg)

    def test_create_tool_helper(self):
        """Verifies standalone create_tool helper function creates EndCallTool."""
        cfg = ToolConfig(name="end_call", type="builtin")
        tool = create_tool(cfg)
        assert isinstance(tool, EndCallTool)
        assert tool.name == "end_call"

    def test_build_tools_default_class_method(self, base_session_config):
        """Verifies ToolRegistry.build_tools_default class method."""
        base_session_config.tools = [ToolConfig(name="end_call", type="builtin")]
        tools = ToolRegistry.build_tools_default(base_session_config)
        assert len(tools) == 1
        assert isinstance(tools[0], EndCallTool)

    def test_registry_build_tools_concurrency_allow_local_isolation(self, base_session_config):
        """Verifies build_tools does not mutate instance-level allow_local state."""
        registry = ToolRegistry(allow_local=False)
        assert registry._allow_local is False

        # Session with allow_local=True in reporting security
        base_session_config.reporting.security.allow_local = True
        base_session_config.tools = [
            ToolConfig(name="webhook_tool", type="webhook", config={"url": "http://127.0.0.1:8080/api"})
        ]

        tools = registry.build_tools(base_session_config)
        assert len(tools) == 1
        # Tool received allow_local=True
        assert tools[0].allow_local is True
        # Registry instance retained allow_local=False
        assert registry._allow_local is False

    def test_registry_custom_builtin_multi_instance_isolation(self):
        """Verifies unregistering a builtin from one registry does not unregister it if another uses it."""
        factory1 = MagicMock()
        factory2 = MagicMock()

        reg1 = ToolRegistry()
        reg2 = ToolRegistry()

        reg1.register_builtin("shared_custom", factory1)
        reg2.register_builtin("shared_custom", factory2)

        # Unregister from reg1
        reg1.unregister_builtin("shared_custom")

        # reg2 still has it registered, and schema still recognizes it
        assert not reg1.is_builtin_registered("shared_custom")
        assert reg2.is_builtin_registered("shared_custom")
        assert ToolConfig(name="shared_custom", type="builtin").name == "shared_custom"

        # Now unregister from reg2
        reg2.unregister_builtin("shared_custom")
        assert not reg2.is_builtin_registered("shared_custom")

    def test_schema_tool_config_drain_timeout_validation(self):
        """Verifies ToolConfig validates positive drain_timeout_seconds."""
        # Valid drain_timeout_seconds
        cfg = ToolConfig(name="end_call", type="builtin", config={"drain_timeout_seconds": 3.5})
        assert cfg.config["drain_timeout_seconds"] == 3.5

        # Invalid zero or negative drain_timeout_seconds
        with pytest.raises(ValueError, match="drain_timeout_seconds' must be a positive number"):
            ToolConfig(name="end_call", type="builtin", config={"drain_timeout_seconds": 0})

        with pytest.raises(ValueError, match="drain_timeout_seconds' must be a positive number"):
            ToolConfig(name="end_call", type="builtin", config={"drain_timeout_seconds": -2.0})
