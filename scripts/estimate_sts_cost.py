#!/usr/bin/env python3
"""Speech-to-Speech (STS) Token Usage & Cost Estimator.

Application-agnostic CLI tool for calculating real-time voice agent costs,
token consumption, WebRTC participant minutes, and monthly volume ledgers.

Supports:
- Loading prompts from arbitrary text files (--prompt-file)
- Loading full SessionConfig metadata JSON blueprints (--metadata-file or --blueprint)
- Dynamic context data template interpolation ({variable} substitution)
- Exact system prompt token counting via Google Gemini API (client.models.count_tokens)
- Audio input & output tokens (at configurable audio token density)
- LiveKit WebRTC participant minutes calculation
- Blended cost per call, cost per continuous hour, and monthly volume ledgers
- Optional live single-turn dialogue test (--test-live-turn)

Usage:
    uv run python scripts/estimate_sts_cost.py
    uv run python scripts/estimate_sts_cost.py --blueprint customer_support --duration 5
    uv run python scripts/estimate_sts_cost.py --prompt-file path/to/prompt.txt --duration 10
    uv run python scripts/estimate_sts_cost.py --metadata-file examples/appointment_booking/booking_metadata.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

# Load environment variables (.env)
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Optional Google GenAI SDK
try:
    from google import genai
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEFAULT PRICING CONSTANTS (GEMINI 3.8 LIVE & LIVEKIT SFU)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gemini 3.8 Live (Speech-to-Speech Multimodal Realtime API)
DEFAULT_PRICE_TEXT_INPUT_PER_1M = 0.75      # $0.75 / 1M text input tokens
DEFAULT_PRICE_AUDIO_INPUT_PER_1M = 3.00     # $3.00 / 1M audio input tokens (~$0.005 / min)
DEFAULT_PRICE_AUDIO_OUTPUT_PER_1M = 12.00   # $12.00 / 1M audio output tokens (~$0.018 / min)
DEFAULT_PRICE_TEXT_OUTPUT_PER_1M = 4.50     # $4.50 / 1M text/thinking tokens

# Gemini Audio Tokenization Rate (Empirical Google API density)
DEFAULT_AUDIO_TOKENS_PER_SECOND = 32        # 32 audio tokens / sec (1,920 tokens / 60 sec)

# LiveKit WebRTC Transport Rates
LIVEKIT_PARTICIPANT_FACTOR = 2.0            # 2 participant minutes per call minute (User + Agent)
LIVEKIT_OVERAGE_PER_PART_MIN = 0.0005       # $0.0005 / participant min = $0.001 / call min

# Built-in Default Prompt (Fallback if no prompt or blueprint specified)
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional, helpful, and concise voice agent assistant. "
    "Listen carefully to the user's questions and provide clear, conversational answers. "
    "Speak conversationally and naturally (1-2 sentences per turn). "
    "Do not use markdown formatting or bullet points as your response is spoken aloud."
)

BLUEPRINT_MAP = {
    "customer_support": PROJECT_ROOT / "examples" / "customer_support" / "support_metadata.json",
    "appointment_booking": PROJECT_ROOT / "examples" / "appointment_booking" / "booking_metadata.json",
    "virtual_receptionist": PROJECT_ROOT / "examples" / "virtual_receptionist" / "receptionist_metadata.json",
}


@dataclass
class CostBreakdown:
    agent_name: str
    prompt_source: str
    duration_minutes: float
    prompt_tokens: int
    user_speech_pct: float
    agent_speech_pct: float
    silence_pct: float
    audio_input_tokens: int
    audio_output_tokens: int
    transcript_tokens: int
    sts_prompt_cost: float
    sts_audio_in_cost: float
    sts_audio_out_cost: float
    sts_transcript_cost: float
    sts_total_cost: float
    livekit_part_mins: float
    livekit_cost: float
    total_cost_per_call: float
    cost_per_hour: float


def get_genai_client() -> Optional[object]:
    """Initializes Google GenAI client if credentials are present."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or not HAS_GOOGLE_GENAI:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def count_prompt_tokens(prompt: str, client: Optional[object] = None, model: str = "gemini-2.0-flash") -> int:
    """Counts tokens via Gemini API, or falls back to UTF-8 heuristic (~4 chars/token)."""
    if client:
        try:
            res = client.models.count_tokens(model=model, contents=prompt)
            return int(res.total_tokens)
        except Exception:
            pass
    # Accurate UTF-8 heuristic: ~3.8 to 4.2 characters per token
    return max(1, int(len(prompt) / 4.0))


def load_prompt_from_metadata(file_path: Path) -> tuple[str, str]:
    """Loads system prompt and agent name from a JSON metadata contract file.

    Uses SessionConfig validation to guarantee canonical spoken dialogue rules
    and dynamic context variable interpolation. Falls back gracefully for partial
    metadata dictionaries.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Attempt standard SessionConfig validation with full interpolation
    try:
        from agent_framework.config.schema import SessionConfig
        config = SessionConfig.model_validate(data)
        return config.get_effective_system_prompt(), config.agent.name
    except Exception:
        pass

    # Fallback extraction for partial metadata dictionaries
    instructions = data.get("instructions", {})
    if isinstance(instructions, dict):
        prompt = instructions.get("system_prompt", "")
        context_data = instructions.get("context_data", {})
        try:
            from agent_framework.utils.templating import interpolate_prompt
            prompt = interpolate_prompt(prompt, context_data)
        except Exception:
            pass

        if instructions.get("spoken_dialogue_rules", True) and "Conversational speech guidelines:" not in prompt:
            prompt = (
                prompt.rstrip()
                + "\n\nConversational speech guidelines:\n"
                + "- Speak conversationally, naturally, and concisely (typically 1-2 sentences per turn).\n"
                + "- Do not format responses with markdown, bullet points, asterisks, or tables, as this will be spoken aloud.\n"
                + "- Ask clarifying questions when needed, but keep them focused and brief."
            )
    elif isinstance(instructions, str):
        prompt = instructions
    else:
        prompt = str(data.get("system_prompt", DEFAULT_SYSTEM_PROMPT))

    agent_name = data.get("agent", {}).get("name", file_path.stem)
    return prompt, agent_name


def calculate_costs(
    prompt: str,
    agent_name: str = "voice-agent",
    prompt_source: str = "custom",
    duration_minutes: float = 5.0,
    user_speech_pct: float = 0.50,
    agent_speech_pct: float = 0.35,
    price_text_in: float = DEFAULT_PRICE_TEXT_INPUT_PER_1M,
    price_audio_in: float = DEFAULT_PRICE_AUDIO_INPUT_PER_1M,
    price_audio_out: float = DEFAULT_PRICE_AUDIO_OUTPUT_PER_1M,
    price_text_out: float = DEFAULT_PRICE_TEXT_OUTPUT_PER_1M,
    audio_tokens_per_sec: int = DEFAULT_AUDIO_TOKENS_PER_SECOND,
    client: Optional[object] = None,
) -> CostBreakdown:
    """Computes exact token metrics and itemized cost breakdown."""
    prompt_tokens = count_prompt_tokens(prompt, client)
    total_seconds = duration_minutes * 60.0
    silence_pct = max(0.0, 1.0 - (user_speech_pct + agent_speech_pct))

    user_speech_seconds = total_seconds * user_speech_pct
    agent_speech_seconds = total_seconds * agent_speech_pct

    audio_input_tokens = int(user_speech_seconds * audio_tokens_per_sec)
    audio_output_tokens = int(agent_speech_seconds * audio_tokens_per_sec)

    # Live transcript text tokens (~140 wpm spoken English * ~1.33 tokens/word)
    total_active_speech_sec = user_speech_seconds + agent_speech_seconds
    total_spoken_words = (total_active_speech_sec / 60.0) * 140.0
    transcript_tokens = int(total_spoken_words * 1.33)

    # Cost calculations
    sts_prompt_cost = (prompt_tokens / 1_000_000.0) * price_text_in
    sts_audio_in_cost = (audio_input_tokens / 1_000_000.0) * price_audio_in
    sts_audio_out_cost = (audio_output_tokens / 1_000_000.0) * price_audio_out
    sts_transcript_cost = (transcript_tokens / 1_000_000.0) * price_text_out
    sts_total_cost = sts_prompt_cost + sts_audio_in_cost + sts_audio_out_cost + sts_transcript_cost

    # LiveKit transport minutes
    livekit_part_mins = duration_minutes * LIVEKIT_PARTICIPANT_FACTOR
    livekit_cost = livekit_part_mins * LIVEKIT_OVERAGE_PER_PART_MIN

    total_cost = sts_total_cost + livekit_cost
    calls_per_hour = (60.0 / duration_minutes) if duration_minutes > 0 else 0
    cost_per_hour = total_cost * calls_per_hour

    return CostBreakdown(
        agent_name=agent_name,
        prompt_source=prompt_source,
        duration_minutes=duration_minutes,
        prompt_tokens=prompt_tokens,
        user_speech_pct=user_speech_pct,
        agent_speech_pct=agent_speech_pct,
        silence_pct=silence_pct,
        audio_input_tokens=audio_input_tokens,
        audio_output_tokens=audio_output_tokens,
        transcript_tokens=transcript_tokens,
        sts_prompt_cost=sts_prompt_cost,
        sts_audio_in_cost=sts_audio_in_cost,
        sts_audio_out_cost=sts_audio_out_cost,
        sts_transcript_cost=sts_transcript_cost,
        sts_total_cost=sts_total_cost,
        livekit_part_mins=livekit_part_mins,
        livekit_cost=livekit_cost,
        total_cost_per_call=total_cost,
        cost_per_hour=cost_per_hour,
    )


def print_report(c: CostBreakdown, price_text_in: float, price_audio_in: float, price_audio_out: float, price_text_out: float):
    """Prints a structured cost and token ledger."""
    total_tokens = c.prompt_tokens + c.audio_input_tokens + c.audio_output_tokens + c.transcript_tokens

    print("\n" + "=" * 80)
    print(f" LIVEKIT STS TOKEN & COST ESTIMATOR -- {c.agent_name.upper()}")
    print("=" * 80)
    print(f" Prompt Source:     {c.prompt_source}")
    print(f" Call Duration:     {c.duration_minutes:.1f} minutes ({c.duration_minutes * 60:.0f} seconds)")
    print(f" Speech Split:      {c.user_speech_pct*100:.0f}% User / {c.agent_speech_pct*100:.0f}% Agent / {c.silence_pct*100:.0f}% Natural Pauses")
    print("=" * 80)

    print("\n[1. TOKEN CONSUMPTION BREAKDOWN]")
    print(f"  - System Prompt (Text In):          {c.prompt_tokens:>8,d} tokens")
    print(f"  - User Audio (Audio In):             {c.audio_input_tokens:>8,d} tokens  ({c.duration_minutes*c.user_speech_pct*60:.0f}s speech)")
    print(f"  - Agent Audio (Audio Out):           {c.audio_output_tokens:>8,d} tokens  ({c.duration_minutes*c.agent_speech_pct*60:.0f}s speech)")
    print(f"  - Live Transcripts (Text Out):       {c.transcript_tokens:>8,d} tokens  (Real-time ASR stream)")
    print(f"  " + "-" * 50)
    print(f"  * TOTAL TOKENS PER CALL:             {total_tokens:>8,d} tokens")

    print("\n[2. ITEMIZED AI MODEL COST PER CALL (USD)]")
    print(f"  - Prompt Text Input:                 ${c.sts_prompt_cost:>8.5f}  (@ ${price_text_in:.2f}/1M tokens)")
    print(f"  - User Audio Input:                  ${c.sts_audio_in_cost:>8.5f}  (@ ${price_audio_in:.2f}/1M tokens = ~$0.005/min)")
    print(f"  - Agent Audio Output:                ${c.sts_audio_out_cost:>8.5f}  (@ ${price_audio_out:.2f}/1M tokens = ~$0.018/min)")
    print(f"  - Transcript Output:                 ${c.sts_transcript_cost:>8.5f}  (@ ${price_text_out:.2f}/1M tokens)")
    print(f"  " + "-" * 50)
    print(f"  * SUB-TOTAL AI MODEL COST:           ${c.sts_total_cost:>8.4f} / call")

    print("\n[3. REAL-TIME MEDIA TRANSPORT (LIVEKIT SFU)]")
    print(f"  - WebRTC Participant Time:           {c.livekit_part_mins:>8.1f} participant mins (2x call duration)")
    print(f"  - WebRTC Bandwidth / SFU Cost:       ${c.livekit_cost:>8.5f}  (@ ${LIVEKIT_OVERAGE_PER_PART_MIN:.4f}/participant min)")

    print("\n[4. BLENDED SUMMARY]")
    print(f"  * TOTAL BLENDED COST PER CALL:       ${c.total_cost_per_call:>8.4f} / call")
    print(f"  * CONTINUOUS OPERATIONAL COST:       ${c.cost_per_hour:>8.2f} / hr of active call time")

    print("\n[5. MONTHLY VOLUME PROJECTIONS]")
    print(f"  +---------------------+------------------+-----------------+------------------+")
    print(f"  | Monthly Call Volume | Total Audio Time | Total AI Tokens |   Blended Cost   |")
    print(f"  +---------------------+------------------+-----------------+------------------+")
    for vol in [100, 500, 1000, 5000, 10000]:
        vol_hrs = (vol * c.duration_minutes) / 60.0
        vol_tokens = vol * total_tokens
        vol_cost = vol * c.total_cost_per_call
        print(f"  | {vol:>7d} calls       | {vol_hrs:>8.1f} hrs      | {vol_tokens:>12,d} |   ${vol_cost:>8.2f} USD  |")
    print(f"  +---------------------+------------------+-----------------+------------------+")
    print("=" * 80)


def test_live_turn(prompt: str, user_turn: str, client: Optional[object], model: str = "gemini-2.0-flash"):
    """Executes a real single-turn test against Gemini using the active prompt."""
    if not client:
        print("\n[!] Cannot run live turn test: GOOGLE_API_KEY is not set or google-genai is missing.")
        return

    print("\n" + "=" * 78)
    print(f" LIVE TURN TEST (Model: {model})")
    print("=" * 78)
    print(f"[User Input Speech]:\n  \"{user_turn}\"")
    print("\nGenerating simulated agent response via Gemini...")

    start_time = time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt, f"User: {user_turn}"],
        )
        elapsed_sec = time.time() - start_time
        reply_text = response.text.strip() if response.text else "[No response]"
        meta_usage = getattr(response, "usage_metadata", None)

        print(f"\n[Agent Response ({elapsed_sec:.2f}s latency)]:")
        print(f"  \"{reply_text}\"")

        print("\n[Actual Gemini Token Usage for This Turn]:")
        if meta_usage:
            print(f"  - Prompt / Input Tokens:         {meta_usage.prompt_token_count:,}")
            print(f"  - Candidates / Output Tokens:    {meta_usage.candidates_token_count:,}")
            if getattr(meta_usage, "thoughts_token_count", None):
                print(f"  - Thinking / Reasoning Tokens:   {meta_usage.thoughts_token_count:,}")
            print(f"  - Total Tokens Transacted:       {meta_usage.total_token_count:,}")
        else:
            print(f"  - Estimated prompt tokens: {len(prompt) // 4:,}")
    except Exception as e:
        print(f"[!] Live turn test encountered an error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate token counts, WebRTC transport, and costs for LiveKit Speech-to-Speech models."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--blueprint",
        choices=list(BLUEPRINT_MAP.keys()),
        help="Use a prebuilt blueprint from examples/ (customer_support, appointment_booking, virtual_receptionist)",
    )
    group.add_argument(
        "--metadata-file",
        type=Path,
        help="Path to a SessionConfig metadata JSON blueprint file",
    )
    group.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a plain text or markdown file containing the system prompt",
    )
    group.add_argument(
        "--prompt-text",
        type=str,
        help="Raw system prompt text string",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Call duration in minutes (default: 5.0)",
    )
    parser.add_argument(
        "--user-speech",
        type=float,
        default=0.50,
        help="Fraction of call time user is speaking (default: 0.50)",
    )
    parser.add_argument(
        "--agent-speech",
        type=float,
        default=0.35,
        help="Fraction of call time agent is speaking (default: 0.35)",
    )
    parser.add_argument(
        "--price-text-in",
        type=float,
        default=DEFAULT_PRICE_TEXT_INPUT_PER_1M,
        help=f"Price per 1M text input tokens (default: ${DEFAULT_PRICE_TEXT_INPUT_PER_1M})",
    )
    parser.add_argument(
        "--price-audio-in",
        type=float,
        default=DEFAULT_PRICE_AUDIO_INPUT_PER_1M,
        help=f"Price per 1M audio input tokens (default: ${DEFAULT_PRICE_AUDIO_INPUT_PER_1M})",
    )
    parser.add_argument(
        "--price-audio-out",
        type=float,
        default=DEFAULT_PRICE_AUDIO_OUTPUT_PER_1M,
        help=f"Price per 1M audio output tokens (default: ${DEFAULT_PRICE_AUDIO_OUTPUT_PER_1M})",
    )
    parser.add_argument(
        "--price-text-out",
        type=float,
        default=DEFAULT_PRICE_TEXT_OUTPUT_PER_1M,
        help=f"Price per 1M text output tokens (default: ${DEFAULT_PRICE_TEXT_OUTPUT_PER_1M})",
    )
    parser.add_argument(
        "--audio-tokens-per-sec",
        type=int,
        default=DEFAULT_AUDIO_TOKENS_PER_SECOND,
        help=f"Audio tokens per second of speech (default: {DEFAULT_AUDIO_TOKENS_PER_SECOND})",
    )
    parser.add_argument(
        "--test-live-turn",
        action="store_true",
        help="Execute a single dialogue turn against Gemini using the active prompt",
    )
    parser.add_argument(
        "--turn-model",
        default="gemini-2.0-flash",
        help="Gemini model for --test-live-turn (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--test-turn-text",
        default="Hello! I need assistance with my account today.",
        help="User speech text used when running --test-live-turn",
    )

    args = parser.parse_args()

    # Input validation
    if args.duration <= 0:
        print(f"[!] Error: Duration must be greater than 0 (got {args.duration})")
        sys.exit(1)

    if args.user_speech < 0 or args.agent_speech < 0:
        print("[!] Error: Speech percentages cannot be negative.")
        sys.exit(1)

    if args.user_speech + args.agent_speech > 1.0:
        print(
            f"[!] Error: Total speech percentage ({args.user_speech + args.agent_speech:.2f}) "
            "cannot exceed 1.0 (100% of call duration)."
        )
        sys.exit(1)

    if any(p < 0 for p in [args.price_text_in, args.price_audio_in, args.price_audio_out, args.price_text_out]):
        print("[!] Error: Token prices cannot be negative.")
        sys.exit(1)

    if args.audio_tokens_per_sec <= 0:
        print(f"[!] Error: audio-tokens-per-sec must be greater than 0 (got {args.audio_tokens_per_sec})")
        sys.exit(1)

    # Determine prompt and agent identity
    agent_name = "voice-agent"
    prompt_source = "default"
    prompt = DEFAULT_SYSTEM_PROMPT

    if args.blueprint:
        meta_path = BLUEPRINT_MAP[args.blueprint]
        try:
            prompt, agent_name = load_prompt_from_metadata(meta_path)
            prompt_source = f"Blueprint: {args.blueprint} ({meta_path.name})"
        except Exception as e:
            print(f"[!] Error loading blueprint '{args.blueprint}': {e}")
            sys.exit(1)
    elif args.metadata_file:
        if not args.metadata_file.exists():
            print(f"[!] Error: Metadata file not found: {args.metadata_file}")
            sys.exit(1)
        try:
            prompt, agent_name = load_prompt_from_metadata(args.metadata_file)
            prompt_source = f"Metadata: {args.metadata_file.name}"
        except json.JSONDecodeError as e:
            print(f"[!] Error: Failed to parse metadata file as JSON: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[!] Error loading metadata file '{args.metadata_file}': {e}")
            sys.exit(1)
    elif args.prompt_file:
        if not args.prompt_file.exists():
            print(f"[!] Error: Prompt file not found: {args.prompt_file}")
            sys.exit(1)
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            agent_name = args.prompt_file.stem
            prompt_source = f"File: {args.prompt_file.name}"
        except Exception as e:
            print(f"[!] Error reading prompt file '{args.prompt_file}': {e}")
            sys.exit(1)
    elif args.prompt_text:
        prompt = args.prompt_text.strip()
        agent_name = "cli-agent"
        prompt_source = "CLI argument"

    client = get_genai_client()
    if client:
        print("[OK] Connected to Google GenAI API (GOOGLE_API_KEY detected). Token counts will be exact.")
    else:
        print("[i] GOOGLE_API_KEY not configured; using high-accuracy UTF-8 character heuristic (~4 chars/token).")

    breakdown = calculate_costs(
        prompt=prompt,
        agent_name=agent_name,
        prompt_source=prompt_source,
        duration_minutes=args.duration,
        user_speech_pct=args.user_speech,
        agent_speech_pct=args.agent_speech,
        price_text_in=args.price_text_in,
        price_audio_in=args.price_audio_in,
        price_audio_out=args.price_audio_out,
        price_text_out=args.price_text_out,
        audio_tokens_per_sec=args.audio_tokens_per_sec,
        client=client,
    )

    print_report(
        breakdown,
        price_text_in=args.price_text_in,
        price_audio_in=args.price_audio_in,
        price_audio_out=args.price_audio_out,
        price_text_out=args.price_text_out,
    )

    if args.test_live_turn:
        test_live_turn(prompt, args.test_turn_text, client, model=args.turn_model)


if __name__ == "__main__":
    main()
