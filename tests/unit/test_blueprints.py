"""Unit tests for Phase 6: Application blueprints, cost estimator, and clean bridges."""

import importlib.util
import json
from pathlib import Path
import pytest

from agent_framework import (
    SessionConfig,
    SessionContext,
    SessionCoordinator,
    create_server,
    main,
    run_app,
    run_server,
    setup_server,
)
import autocoach

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Dynamically import scripts/estimate_sts_cost.py
estimate_script_path = PROJECT_ROOT / "scripts" / "estimate_sts_cost.py"
spec = importlib.util.spec_from_file_location("estimate_sts_cost", estimate_script_path)
estimate_sts_cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimate_sts_cost)

BLUEPRINT_MAP = estimate_sts_cost.BLUEPRINT_MAP
CostBreakdown = estimate_sts_cost.CostBreakdown
calculate_costs = estimate_sts_cost.calculate_costs
count_prompt_tokens = estimate_sts_cost.count_prompt_tokens
load_prompt_from_metadata = estimate_sts_cost.load_prompt_from_metadata


class TestApplicationBlueprints:
    """Verifies that all example blueprints in examples/ strictly conform to SessionConfig."""

    def test_all_blueprints_valid_schema(self):
        """Verifies every JSON blueprint in examples/ parses and validates under SessionConfig."""
        examples_dir = PROJECT_ROOT / "examples"
        json_files = list(examples_dir.rglob("*.json"))
        assert len(json_files) >= 3, f"Expected at least 3 blueprints, found {len(json_files)}"

        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            config = SessionConfig.model_validate(data)
            assert config.version == "1.0"
            assert config.models.provider in {"gemini", "openai"}
            assert config.voice.voice_id
            assert config.instructions.system_prompt
            assert len(config.tools) >= 1

    def test_customer_support_blueprint(self):
        """Verifies customer_support blueprint structure and tools."""
        path = PROJECT_ROOT / "examples" / "customer_support" / "support_metadata.json"
        assert path.exists()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = SessionConfig.model_validate(data)

        assert config.agent.name == "support-tier1-agent"
        assert config.agent.role == "technical_support"
        assert config.models.model == "gemini-3.8-live"
        assert config.models.thinking.enabled is True
        assert config.instructions.first_turn.speaker == "agent"
        assert config.instructions.first_turn.mode == "say"
        assert "CloudServices" in config.instructions.first_turn.text

        tool_names = [t.name for t in config.tools]
        assert "lookup_account" in tool_names
        assert "end_call" in tool_names

        lookup_tool = next(t for t in config.tools if t.name == "lookup_account")
        assert lookup_tool.type == "webhook"
        assert lookup_tool.config["url"] == "https://api.example.com/v1/support/accounts/lookup"

        assert config.reporting.enabled is True
        assert config.reporting.security.validate_ssrf is True

    def test_appointment_booking_blueprint(self):
        """Verifies appointment_booking blueprint structure and tools."""
        path = PROJECT_ROOT / "examples" / "appointment_booking" / "booking_metadata.json"
        assert path.exists()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = SessionConfig.model_validate(data)

        assert config.agent.name == "universal-booking-agent"
        assert config.agent.role == "scheduler"
        assert config.voice.voice_id == "Puck"

        tool_names = [t.name for t in config.tools]
        assert "check_availability" in tool_names
        assert "book_appointment" in tool_names
        assert "end_call" in tool_names

        availability_tool = next(t for t in config.tools if t.name == "check_availability")
        assert availability_tool.type == "webhook"
        assert "service_type" in availability_tool.parameters["required"]

    def test_virtual_receptionist_blueprint(self):
        """Verifies virtual_receptionist blueprint structure and greeting."""
        path = PROJECT_ROOT / "examples" / "virtual_receptionist" / "receptionist_metadata.json"
        assert path.exists()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = SessionConfig.model_validate(data)

        assert config.agent.name == "inbound-receptionist-agent"
        assert config.agent.role == "receptionist"
        assert config.instructions.first_turn.speaker == "agent"
        assert "Horizon Tower" in config.instructions.first_turn.text
        assert len(config.tools) == 1
        assert config.tools[0].name == "end_call"

    def test_context_interpolation_in_all_blueprints(self):
        """Verifies context_data is dynamically interpolated into the effective system prompt."""
        # 1. Customer support interpolates {portal_url}
        support_path = PROJECT_ROOT / "examples" / "customer_support" / "support_metadata.json"
        with open(support_path, "r", encoding="utf-8") as f:
            support_data = json.load(f)
        support_config = SessionConfig.model_validate(support_data)
        effective_support = support_config.get_effective_system_prompt()
        assert "https://portal.cloudservices.example.com" in effective_support
        assert "{portal_url}" not in effective_support

        # 2. Appointment booking interpolates {business_name} and {timezone}
        booking_path = PROJECT_ROOT / "examples" / "appointment_booking" / "booking_metadata.json"
        with open(booking_path, "r", encoding="utf-8") as f:
            booking_data = json.load(f)
        booking_config = SessionConfig.model_validate(booking_data)
        effective_booking = booking_config.get_effective_system_prompt()
        assert "Metro Wellness Center" in effective_booking
        assert "America/New_York" in effective_booking
        assert "{business_name}" not in effective_booking

        # 3. Virtual receptionist interpolates {address} and {building_hours}
        receptionist_path = PROJECT_ROOT / "examples" / "virtual_receptionist" / "receptionist_metadata.json"
        with open(receptionist_path, "r", encoding="utf-8") as f:
            receptionist_data = json.load(f)
        receptionist_config = SessionConfig.model_validate(receptionist_data)
        effective_receptionist = receptionist_config.get_effective_system_prompt()
        assert "100 Horizon Boulevard, Suite 500" in effective_receptionist
        assert "Monday through Friday from 7 AM to 7 PM" in effective_receptionist
        assert "{address}" not in effective_receptionist


class TestCostEstimatorGeneralization:
    """Verifies that scripts/estimate_sts_cost.py functions without legacy dependencies."""

    def test_count_prompt_tokens_heuristic(self):
        """Verifies character-based heuristic token counting when client is None."""
        text = "Hello world! This is a test prompt for estimating token usage."
        tokens = count_prompt_tokens(text, client=None)
        assert tokens > 0
        assert tokens == max(1, int(len(text) / 4.0))

    def test_load_prompt_from_metadata_all_blueprints(self):
        """Verifies load_prompt_from_metadata extracts prompts from all example blueprints."""
        for name, path in BLUEPRINT_MAP.items():
            prompt, agent_name = load_prompt_from_metadata(path)
            assert isinstance(prompt, str) and len(prompt) > 20
            assert isinstance(agent_name, str) and len(agent_name) > 0
            assert "Conversational speech guidelines:" in prompt

    def test_load_prompt_missing_file_raises(self):
        """Verifies load_prompt_from_metadata raises FileNotFoundError for missing paths."""
        with pytest.raises(FileNotFoundError):
            load_prompt_from_metadata(PROJECT_ROOT / "non_existent_metadata.json")

    def test_load_prompt_interpolates_context(self):
        """Verifies load_prompt_from_metadata properly interpolates context_data."""
        path = PROJECT_ROOT / "examples" / "customer_support" / "support_metadata.json"
        prompt, agent_name = load_prompt_from_metadata(path)
        assert "https://portal.cloudservices.example.com" in prompt
        assert "{portal_url}" not in prompt

    def test_calculate_costs_output(self):
        """Verifies calculate_costs produces a complete and valid CostBreakdown."""
        prompt = "You are a test voice agent assistant."
        breakdown = calculate_costs(
            prompt=prompt,
            agent_name="test-agent",
            prompt_source="unit-test",
            duration_minutes=5.0,
            user_speech_pct=0.50,
            agent_speech_pct=0.35,
        )

        assert isinstance(breakdown, CostBreakdown)
        assert breakdown.agent_name == "test-agent"
        assert breakdown.duration_minutes == 5.0
        assert breakdown.audio_input_tokens > 0
        assert breakdown.audio_output_tokens > 0
        assert breakdown.transcript_tokens > 0
        assert breakdown.sts_total_cost > 0
        assert breakdown.livekit_part_mins == 10.0
        assert breakdown.total_cost_per_call > 0
        assert breakdown.cost_per_hour == breakdown.total_cost_per_call * 12.0


class TestAutocoachBridgeAndServerMain:
    """Verifies the backward compatibility re-export bridge and CLI entrypoints."""

    def test_autocoach_reexports(self):
        """Verifies autocoach module re-exports agent_framework symbols."""
        assert autocoach.SessionConfig is SessionConfig
        assert autocoach.SessionContext is SessionContext
        assert autocoach.SessionCoordinator is SessionCoordinator
        assert autocoach.create_server is create_server
        assert autocoach.setup_server is setup_server
        assert autocoach.run_server is run_server
        assert autocoach.run_app is run_app
        assert autocoach.main is main

    def test_server_main_callable(self):
        """Verifies that the main function exposed in pyproject.toml is callable."""
        assert callable(main)

    def test_package_main_modules(self):
        """Verifies __main__.py in both agent_framework and autocoach import main."""
        import agent_framework.__main__
        import autocoach.__main__
        assert agent_framework.__main__.main is main
        assert autocoach.__main__.main is main
