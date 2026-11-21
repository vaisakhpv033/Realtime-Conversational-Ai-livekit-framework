# LiveKit Voice Agent Framework (Gemini 3.8 Live STS)

A pure greenfield, application-agnostic Speech-to-Speech (STS) voice agent framework built with **LiveKit Agents** and **Google Gemini 3.8 Live** (`livekit-plugins-google`, `google-genai`).

Designed to run low-latency, bidirectional, multi-tenant voice agents where **100% of the agent's behavior, instructions, voice acoustics, models, tools, and telemetry are dynamically controlled via incoming LiveKit metadata**.

---

## Key Capabilities

- **Pure Application Independence**: Zero hardcoded personas, zero domain prompt templates, and zero application assumptions. Built to serve technical support, appointments, triage, coaching, or general concierge workflows out-of-the-box.
- **Gemini 3.8 Live Native STS**: True real-time bidirectional speech-to-speech audio streaming over WebSockets without separate STT/TTS latency penalties.
- **Strict Pydantic V2 Contract**: 10-section fail-fast validation (`SessionConfig`) ensuring missing or malformed fields are rejected immediately with structured diagnostics.
- **50,000-Token Sliding Window**: Built-in `ContextWindowCompressionConfig` preventing session cutoff and WebSocket 1007 crashes on calls exceeding typical audio limits.
- **Turn-Taking & Interruption Defense**: LiveKit `TurnHandlingOptions` configured with `resume_false_interruption=True` and locked ASR language tags to prevent speech truncation during backchanneling ("uh-huh", "yeah") and interim transcription language drift.
- **Dynamic Tool Registry**: Dynamic function execution supporting both REST webhook tools (with secure `{{env:KEY}}` header resolution) and graceful call termination (`end_call`) with audio drain synchronization.
- **Resilient Reporting & SSRF Hardening**: Asynchronous post-call telemetry dispatching with pre-flight DNS SSRF validation (blocking cloud metadata IPs `169.254.169.254` and loopbacks) and 3-attempt exponential backoff.
- **Multi-Tenant Process Isolation**: Scoped `SessionContext` bound per LiveKit `JobContext` with zero module-level state leakage across concurrent worker sessions.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |          Caller / Human User          |
                      |   (Browser / SIP Telephony / WebRTC)  |
                      +-------------------+-------------------+
                                          | WebRTC Audio (Opus / PCM)
                                          v
                      +---------------------------------------+
                      |         LiveKit SFU / Cloud           |
                      |   (Room Metadata & Job Dispatching)   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     LiveKit Agent Framework Worker    |
                      |    (SessionCoordinator on Docker)     |
                      +---------+-------------------+---------+
                                |                   |
    Bidirectional Audio / State |                   | Dynamic Telemetry
       (16kHz in / 24kHz out)   |                   | (SSRF Guarded Webhook)
                                v                   v
             +-----------------------+   +-----------------------+
             | Google Gemini 3.8 Live|   |   Your Backend API /  |
             |  (or OpenAI Realtime) |   |   Analytics Webhook   |
             +-----------------------+   +-----------------------+
```

---

## Directory Structure

```
Autocoach-agent/
├── pyproject.toml              # Dependencies, uv build, and CLI entrypoint
├── uv.lock                     # Deterministic dependency lockfile
├── Dockerfile                  # Multi-stage production container build
├── docker-compose.yml          # Self-hosted stack with port 8081 health checks
├── .env.example                # Environment configuration template
├── examples/                   # Clean, domain-neutral application blueprints
│   ├── customer_support/       # Tier 1 Support Agent with REST account lookup
│   ├── appointment_booking/    # Universal Booking Agent with calendar tools
│   ├── virtual_receptionist/   # Inbound Concierge with greeting utterance
│   └── README.md               # LiveKit CLI and Backend API dispatch guide
├── scripts/
│   └── estimate_sts_cost.py    # General CLI for token & WebRTC cost calculation
├── src/
│   ├── agent_framework/        # Core framework package
│   │   ├── config/             # Pydantic V2 schema, validator, resolver, settings
│   │   ├── core/               # Session coordinator, context, server runner
│   │   ├── models/             # Gemini Live adapter & model factory
│   │   ├── reporting/          # SSRF security, payload assembly, dispatcher
│   │   ├── tools/              # Dynamic tool registry, builtin end_call, webhooks
│   │   ├── turn_detection/     # VAD and conversational dynamics factory
│   │   └── utils/              # Secret resolution and template interpolation
│   └── autocoach/              # Backward-compatibility re-export bridge
└── tests/
    └── unit/                   # 210+ comprehensive unit test suite
```

---

## Quick Start (Local Development)

### 1. Prerequisites
- Python 3.13
- [uv package manager](https://github.com/astral-sh/uv)

### 2. Installation
```bash
# Clone the repository
git clone <repo-url>
cd Autocoach-agent

# Install dependencies into virtual environment
uv sync
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Ensure you provide:
- `LIVEKIT_URL`: LiveKit server WebSocket URL (`wss://...`)
- `LIVEKIT_API_KEY`: LiveKit project API key
- `LIVEKIT_API_SECRET`: LiveKit project API secret
- `GOOGLE_API_KEY`: Google Gemini API key

### 4. Run Worker
Start the worker locally:
```bash
uv run agent-framework start
```
Or in dev mode with hot reload:
```bash
uv run agent-framework dev
```

---

## Dispatching Prebuilt Blueprints

The `examples/` directory contains complete metadata contracts ready for dispatch:

1. **Tier 1 Customer Support** (`examples/customer_support/support_metadata.json`): Technical troubleshooting, billing questions, thinking model enabled, and REST lookup tool.
2. **Appointment Booking** (`examples/appointment_booking/booking_metadata.json`): Calendar availability checking and booking confirmation tools with dynamic context interpolation.
3. **Virtual Receptionist** (`examples/virtual_receptionist/receptionist_metadata.json`): Inbound front-desk concierge with initial greeting utterance.

### Dispatch via LiveKit CLI (`lk`)
```bash
lk room create \
  --name "support-demo" \
  --metadata "$(cat examples/customer_support/support_metadata.json)"
```

---

## Estimating STS Token & Infrastructure Costs

The CLI utility `scripts/estimate_sts_cost.py` calculates exact token usage, LiveKit SFU participant minutes, and blended cost per call:

```bash
# Estimate customer support blueprint (5-minute call)
uv run python scripts/estimate_sts_cost.py --blueprint customer_support --duration 5

# Estimate appointment booking blueprint (3-minute call)
uv run python scripts/estimate_sts_cost.py --blueprint appointment_booking --duration 3

# Estimate arbitrary prompt file
uv run python scripts/estimate_sts_cost.py --prompt-file path/to/prompt.txt --duration 10
```

---

## Running Automated Tests

Run the full pytest suite:
```bash
uv run pytest tests/
```
All unit tests in `tests/unit/` run cleanly without requiring an active LiveKit server or external API quota.
