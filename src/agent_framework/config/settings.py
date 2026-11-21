"""Infrastructure and runtime configuration settings for the LiveKit Agent Framework.

Provides strongly-typed environment settings for LiveKit SFU connectivity,
health check endpoints, worker shutdown timeouts, and language code resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Locate project root directory and optional .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


# Standard mapping of ISO language identifiers and colloquial names to BCP-47 tags
LANGUAGE_BCP47_MAP: dict[str, str] = {
    "en": "en-US",
    "en-us": "en-US",
    "en-in": "en-IN",
    "en-gb": "en-GB",
    "english": "en-US",
    "es": "es-US",
    "es-us": "es-US",
    "es-es": "es-ES",
    "es-mx": "es-MX",
    "spanish": "es-US",
    "hi": "hi-IN",
    "hindi": "hi-IN",
    "ml": "ml-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "ur": "ur-IN",
    "fr": "fr-FR",
    "french": "fr-FR",
    "de": "de-DE",
    "german": "de-DE",
    "it": "it-IT",
    "italian": "it-IT",
    "pt": "pt-BR",
    "portuguese": "pt-BR",
    "zh": "zh-CN",
    "chinese": "zh-CN",
    "ja": "ja-JP",
    "japanese": "ja-JP",
    "ko": "ko-KR",
    "korean": "ko-KR",
    "ar": "ar-AE",
    "arabic": "ar-AE",
}


class FrameworkSettings(BaseSettings):
    """Infrastructure settings for worker processes and SFU connectivity."""

    model_config = SettingsConfigDict(
        env_file=(".env", str(ENV_PATH)),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LiveKit SFU Connection Credentials
    livekit_url: str = Field(default="ws://127.0.0.1:7880", alias="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", alias="LIVEKIT_API_SECRET")

    # Model Provider API Credentials
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Worker Infrastructure & Server Settings
    agent_name: str = Field(default="livekit-voice-agent", alias="AGENT_NAME")
    health_check_host: str = Field(default="0.0.0.0", alias="HEALTH_CHECK_HOST")
    health_check_port: int = Field(default=8081, alias="HEALTH_CHECK_PORT")
    shutdown_timeout_seconds: float = Field(default=60.0, alias="SHUTDOWN_TIMEOUT_SECONDS")
    worker_load_threshold: float = Field(default=0.7, alias="WORKER_LOAD_THRESHOLD")
    allow_local_webhooks: bool = Field(default=False, alias="ALLOW_LOCAL_WEBHOOKS")

    def resolve_bcp47_language(self, language_code: Optional[str]) -> str:
        """Resolves a raw language code or common language name into a verified BCP-47 tag."""
        if not language_code:
            return "en-US"
        clean = language_code.strip().lower()
        if clean in LANGUAGE_BCP47_MAP:
            return LANGUAGE_BCP47_MAP[clean]
        # Fallback to base ISO code (e.g. en_US -> en -> en-US)
        base_code = clean.split("-")[0].split("_")[0]
        return LANGUAGE_BCP47_MAP.get(base_code, "en-US")

    def resolve_asr_language_codes(
        self, language_code: Optional[str], extra_languages: Optional[list[str]] = None
    ) -> list[str]:
        """Resolves target ASR language codes to lock speech-to-text recognition,
        preventing interim language-jumping hallucinations.
        """
        primary = self.resolve_bcp47_language(language_code)
        codes = [primary]
        if extra_languages:
            for extra in extra_languages:
                resolved = self.resolve_bcp47_language(extra)
                if resolved not in codes:
                    codes.append(resolved)
        return codes


# Global singleton settings instance
settings = FrameworkSettings()
