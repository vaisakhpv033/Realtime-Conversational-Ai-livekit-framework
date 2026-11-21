"""Turn-detection and speech dynamics factory for LiveKit AgentSession.

Translates SessionConfig turn detection parameters into LiveKit 1.8+ TurnHandlingOptions,
InterruptionOptions, and EndpointingOptions.
"""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import TurnHandlingOptions
from livekit.agents.voice.turn import EndpointingOptions, InterruptionOptions

from agent_framework.config.schema import SessionConfig

logger = logging.getLogger("agent_framework.turn_detection.factory")


class TurnDetectionFactory:
    """Factory translating SessionConfig turn detection settings into LiveKit structures."""

    @classmethod
    def build_turn_handling_options(
        cls, session_config: SessionConfig
    ) -> TurnHandlingOptions:
        """Builds TurnHandlingOptions from validated SessionConfig.

        Maps:
        - session_config.turn_detection.allow_interruptions -> interruption['enabled']
        - session_config.turn_detection.resume_false_interruption -> interruption['resume_false_interruption']
        - session_config.turn_detection.min_interruption_duration_ms -> interruption['min_duration'] (seconds)
        - session_config.turn_detection.min_interruption_words -> interruption['min_words']
        - session_config.turn_detection.min_endpointing_delay_ms -> endpointing['min_delay'] (seconds)

        Args:
            session_config: Validated session configuration.

        Returns:
            A TurnHandlingOptions dict suitable for AgentSession(turn_handling=...).
        """
        td = session_config.turn_detection

        # Convert millisecond thresholds to seconds for LiveKit
        min_interruption_duration_sec = td.min_interruption_duration_ms / 1000.0
        min_endpointing_delay_sec = td.min_endpointing_delay_ms / 1000.0

        interruption = InterruptionOptions(
            enabled=td.allow_interruptions,
            resume_false_interruption=td.resume_false_interruption,
            min_duration=min_interruption_duration_sec,
            min_words=td.min_interruption_words,
        )

        endpointing = EndpointingOptions(
            min_delay=min_endpointing_delay_sec,
        )

        turn_handling = TurnHandlingOptions(
            interruption=interruption,
            endpointing=endpointing,
        )

        if td.mode in {"realtime_llm", "vad", "stt", "manual"}:
            turn_handling["turn_detection"] = td.mode

        logger.debug(
            f"Configured TurnHandlingOptions: interruptions={td.allow_interruptions}, "
            f"resume_false={td.resume_false_interruption}, "
            f"min_duration={min_interruption_duration_sec}s, min_words={td.min_interruption_words}, "
            f"min_endpointing_delay={min_endpointing_delay_sec}s"
        )
        return turn_handling

    @classmethod
    def get_user_away_timeout(cls, session_config: SessionConfig) -> float:
        """Returns the user-away inactivity timeout in seconds from SessionConfig.

        Args:
            session_config: Validated session configuration.

        Returns:
            Inactivity timeout in seconds as a float.
        """
        return float(session_config.turn_detection.user_away_timeout_seconds)
