"""Dynamic REST API tool executing external endpoints via LiveKit's FunctionTool."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Mapping, Optional, Union
import urllib.parse
import aiohttp
from livekit.agents.llm import RawFunctionTool, Toolset, function_tool

from agent_framework.config.schema import ToolConfig
from agent_framework.reporting.security import is_safe_webhook_url
from agent_framework.utils.secrets import resolve_secret

logger = logging.getLogger("agent_framework.tools.webhook_tool")


class WebhookToolError(Exception):
    """Base exception for WebhookTool failures."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SSRFSecurityError(WebhookToolError):
    """Raised when a webhook target URL violates SSRF security policies."""
    pass


class WebhookHTTPError(WebhookToolError):
    """Raised when an HTTP endpoint returns a 4xx or 5xx status code."""
    pass


class WebhookTimeoutError(WebhookToolError):
    """Raised when a webhook request times out."""
    pass


class WebhookConnectionError(WebhookToolError):
    """Raised on network or connection errors."""
    pass


class WebhookTool(Toolset):
    """Dynamic tool executing external REST APIs on behalf of the voice model.

    Features:
    - Built from JSON Schema parameter definitions matching OpenAPI / JSON schema standards.
    - Asynchronous HTTP dispatch via aiohttp.ClientSession (supporting GET, POST, PUT).
    - Secret resolution in headers and URL via {{env:KEY}} and env:KEY templates.
    - URL path template parameter substitution (e.g., https://api.com/users/{id}).
    - SSRF pre-dispatch validation via is_safe_webhook_url blocking metadata IPs and private ranges.
    - Graceful error handling returning structured JSON strings back to the model without crashing.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        description: str = "",
        parameters: Optional[dict[str, Any]] = None,
        url: Optional[str] = None,
        method: str = "POST",
        headers: Optional[dict[str, str]] = None,
        timeout_seconds: float = 5.0,
        allow_local: bool = False,
        allowed_domains: Optional[list[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        tool_config: Optional[ToolConfig] = None,
        session: Optional[aiohttp.ClientSession] = None,
        raise_on_error: bool = False,
        max_response_chars: int = 50_000,
    ) -> None:
        """Initializes the WebhookTool.

        Args:
            name: Name of the tool exposed to the LLM.
            description: Description of the tool exposed to the LLM.
            parameters: Standard JSON Schema describing tool argument properties.
            url: Target webhook HTTP URL.
            method: HTTP method ('GET', 'POST', 'PUT').
            headers: Custom request headers (supports {{env:KEY}} references).
            timeout_seconds: Request timeout in seconds.
            allow_local: Allows loopback/private IPs (for testing only).
            allowed_domains: Optional list of whitelisted domain names/suffixes.
            env: Optional environment mapping for secret resolution.
            tool_config: Optional ToolConfig instance to populate configuration.
            session: Optional pre-existing aiohttp.ClientSession.
            raise_on_error: If True, raises exceptions instead of returning error JSON.
            max_response_chars: Maximum characters to retain from response bodies.
        """
        if tool_config is not None:
            self._name = name or tool_config.name
            self._description = description or tool_config.description
            self._parameters = parameters if parameters is not None else tool_config.parameters
            cfg = tool_config.config or {}
            self._url = url or cfg.get("url", "")
            self._method = (method if method != "POST" else cfg.get("method", "POST")).upper()
            self._headers = headers if headers is not None else cfg.get("headers", {})
            self._timeout_seconds = timeout_seconds if timeout_seconds != 5.0 else float(cfg.get("timeout_seconds", 5.0))
            self._max_response_chars = max(100, int(cfg.get("max_response_chars", max_response_chars)))
        else:
            self._name = (name or "").strip()
            self._description = description
            self._parameters = parameters or {}
            self._url = (url or "").strip()
            self._method = (method or "POST").strip().upper()
            self._headers = dict(headers or {})
            self._timeout_seconds = float(timeout_seconds)
            self._max_response_chars = max(100, int(max_response_chars))

        if not self._name:
            raise ValueError("WebhookTool name cannot be empty")
        if not self._url:
            raise ValueError(f"WebhookTool '{self._name}' must specify a target URL")
        if self._method not in ("GET", "POST", "PUT"):
            raise ValueError(f"Invalid HTTP method '{self._method}'. Supported: GET, POST, PUT")

        self._allow_local = allow_local
        self._allowed_domains = allowed_domains
        self._env = env
        self._session = session
        self._raise_on_error = raise_on_error

        # Ensure parameters has valid JSON Schema structure
        schema_params = self._parameters if isinstance(self._parameters, dict) and "type" in self._parameters else {
            "type": "object",
            "properties": self._parameters or {},
        }

        raw_schema = {
            "name": self._name,
            "description": self._description,
            "parameters": schema_params,
        }

        # Handler function with explicit type hints required by LiveKit prepare_function_arguments
        async def _raw_tool_handler(raw_arguments: dict[str, Any], **extra: Any) -> str:
            return await self.execute(raw_arguments)

        fn_tool = function_tool(raw_schema=raw_schema)(_raw_tool_handler)
        super().__init__(id=self._name, tools=[fn_tool])
        self._fn_tool = self.tools[0]

    @classmethod
    def from_config(
        cls,
        tool_config: ToolConfig,
        *,
        env: Optional[Mapping[str, str]] = None,
        allow_local: bool = False,
        allowed_domains: Optional[list[str]] = None,
        session: Optional[aiohttp.ClientSession] = None,
        raise_on_error: bool = False,
        max_response_chars: int = 50_000,
    ) -> WebhookTool:
        """Constructs a WebhookTool from a validated ToolConfig schema model."""
        return cls(
            tool_config=tool_config,
            env=env,
            allow_local=allow_local,
            allowed_domains=allowed_domains,
            session=session,
            raise_on_error=raise_on_error,
            max_response_chars=max_response_chars,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def url(self) -> str:
        return self._url

    @property
    def method(self) -> str:
        return self._method

    @property
    def headers(self) -> dict[str, str]:
        return self._headers

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def allow_local(self) -> bool:
        return self._allow_local

    @property
    def allowed_domains(self) -> Optional[list[str]]:
        return self._allowed_domains

    @property
    def raise_on_error(self) -> bool:
        return self._raise_on_error

    @property
    def max_response_chars(self) -> int:
        return self._max_response_chars

    @property
    def function_tool(self) -> RawFunctionTool:
        """Returns the underlying LiveKit RawFunctionTool instance."""
        return self._fn_tool

    def to_function_tool(self) -> RawFunctionTool:
        """Convenience accessor returning the LiveKit function tool."""
        return self._fn_tool

    async def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Permits direct invocation: await tool(arg1=val1) or await tool({'arg1': 'val1'})."""
        return await self.execute(*args, **kwargs)

    async def execute(self, *args: Any, **kwargs: Any) -> str:
        """Executes the HTTP request against the configured endpoint.

        Returns:
            JSON-serialized string response intended for LLM consumption.
        """
        # 1. Parse and extract arguments
        if "raw_arguments" in kwargs:
            raw_args = kwargs["raw_arguments"]
        elif args and isinstance(args[0], dict):
            raw_args = args[0]
        else:
            raw_args = kwargs

        args_dict = dict(raw_args) if isinstance(raw_args, dict) else {}

        # 2. Resolve target URL (and interpolate path parameters)
        target_url = resolve_secret(self._url, env=self._env, strict=False)
        remaining_args = dict(args_dict)

        for key, val in list(args_dict.items()):
            # Support both {key} and {{key}} URL path variable interpolation with URL quoting
            pattern = re.compile(r"\{+" + re.escape(str(key)) + r"\}+")
            if pattern.search(target_url):
                encoded_val = urllib.parse.quote(str(val), safe="")
                target_url = pattern.sub(encoded_val, target_url)
                remaining_args.pop(key, None)

        # Check for unreplaced path variables in URL
        unreplaced = re.findall(r"\{+([a-zA-Z0-9_]+)\}+", target_url)
        if unreplaced:
            msg = f"Missing required URL path parameter(s): {', '.join(unreplaced)}"
            logger.error(f"[{self._name}] {msg}")
            if self._raise_on_error:
                raise WebhookToolError(msg, status_code=400)
            return json.dumps({
                "error": msg,
                "status": "error",
                "status_code": 400,
                "url": target_url,
            })

        # 3. SSRF Validation
        is_safe, ssrf_reason = is_safe_webhook_url(
            target_url,
            allow_local=self._allow_local,
            allowed_domains=self._allowed_domains,
        )
        if not is_safe:
            msg = f"SSRF security check rejected webhook target '{target_url}': {ssrf_reason}"
            logger.error(f"[{self._name}] {msg}")
            if self._raise_on_error:
                raise SSRFSecurityError(msg, status_code=400)
            return json.dumps({
                "error": msg,
                "status": "blocked",
                "status_code": 400,
                "url": target_url,
            })

        # 4. Resolve headers and secrets (case-insensitive defaults)
        lower_headers = {k.lower(): k for k in self._headers.keys()}
        resolved_headers: dict[str, str] = {}
        if "user-agent" not in lower_headers:
            resolved_headers["User-Agent"] = "AgentFramework-WebhookTool/1.0"
        if self._method in ("POST", "PUT") and "content-type" not in lower_headers:
            resolved_headers["Content-Type"] = "application/json"

        for k, v in self._headers.items():
            try:
                resolved_headers[k] = resolve_secret(str(v), env=self._env, strict=True)
            except Exception as e:
                msg = f"Failed to resolve secret in header '{k}': {e}"
                logger.error(f"[{self._name}] {msg}")
                if self._raise_on_error:
                    raise WebhookToolError(msg, status_code=500)
                return json.dumps({
                    "error": msg,
                    "status_code": 500,
                })

        # 5. Execute HTTP Request
        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        owns_session = False
        client_session = self._session

        if client_session is None or client_session.closed:
            client_session = aiohttp.ClientSession()
            owns_session = True

        try:
            logger.debug(f"[{self._name}] Dispatching {self._method} request to {target_url}")

            if self._method == "GET":
                # Normalize query parameters so primitives, booleans, sequences, and nested structures serialize cleanly
                query_params: dict[str, Any] = {}
                for k, v in remaining_args.items():
                    if isinstance(v, bool):
                        query_params[k] = "true" if v else "false"
                    elif isinstance(v, (int, float, str)):
                        query_params[k] = str(v)
                    elif isinstance(v, (list, tuple)):
                        if all(isinstance(x, (int, float, str, bool)) for x in v):
                            query_params[k] = [
                                "true" if x is True else ("false" if x is False else str(x))
                                for x in v
                            ]
                        else:
                            query_params[k] = json.dumps(v)
                    elif v is not None:
                        query_params[k] = json.dumps(v)

                req_coro = client_session.get(
                    target_url,
                    headers=resolved_headers,
                    params=query_params,
                    timeout=timeout,
                )
            else:
                req_coro = client_session.request(
                    self._method,
                    target_url,
                    headers=resolved_headers,
                    json=remaining_args,
                    timeout=timeout,
                )

            async with req_coro as resp:
                resp_text = await resp.text()
                status = resp.status

                # Truncate text if it exceeds maximum character limit
                is_truncated = len(resp_text) > self._max_response_chars
                safe_resp_text = resp_text[: self._max_response_chars]
                if is_truncated:
                    safe_resp_text += "... [truncated]"

                # Handle HTTP errors (4xx, 5xx)
                if status >= 400:
                    msg = f"Webhook request to '{target_url}' returned HTTP {status}"
                    logger.warning(f"[{self._name}] {msg}: {safe_resp_text[:200]}")
                    if self._raise_on_error:
                        raise WebhookHTTPError(f"{msg}: {safe_resp_text}", status_code=status)
                    return json.dumps({
                        "error": msg,
                        "status_code": status,
                        "response": safe_resp_text,
                    })

                # Successful HTTP response: parse as JSON if possible
                try:
                    data = json.loads(resp_text)
                    return json.dumps(data)
                except Exception:
                    return json.dumps({
                        "response": safe_resp_text,
                        "status_code": status,
                    })

        except asyncio.TimeoutError as e:
            msg = f"Webhook request to '{target_url}' timed out after {self._timeout_seconds}s"
            logger.warning(f"[{self._name}] {msg}")
            if self._raise_on_error:
                raise WebhookTimeoutError(msg, status_code=504) from e
            return json.dumps({
                "error": msg,
                "status_code": 504,
            })

        except aiohttp.ClientError as e:
            msg = f"Webhook network error connecting to '{target_url}': {e}"
            logger.warning(f"[{self._name}] {msg}")
            if self._raise_on_error:
                raise WebhookConnectionError(msg, status_code=502) from e
            return json.dumps({
                "error": msg,
                "status_code": 502,
            })

        except WebhookToolError:
            raise

        except Exception as e:
            msg = f"Unexpected error during webhook execution: {e}"
            logger.exception(f"[{self._name}] {msg}")
            if self._raise_on_error:
                raise WebhookToolError(msg, status_code=500) from e
            return json.dumps({
                "error": msg,
                "status_code": 500,
            })

        finally:
            if owns_session and not client_session.closed:
                await client_session.close()
