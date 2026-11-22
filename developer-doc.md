# LiveKit Voice Agent Framework - Developer Documentation & Metadata Contract Reference

A comprehensive architectural, integration, and configuration manual for engineers building and deploying real-time Speech-to-Speech (STS) voice agents using the **LiveKit Voice Agent Framework** powered by **Google Gemini 3.8 Live** (`livekit-agents` 1.8+, `livekit-plugins-google` 1.8.5+, and `google-genai` 2.10+).

---

## Table of Contents

1. [Executive Overview & Framework Architecture](#1-executive-overview-framework-architecture)
   - [1.1 What is this Agent Framework?](#11-what-is-this-agent-framework)
   - [1.2 Speech-to-Speech (STS) Pipeline vs. Cascaded Architectures](#12-speech-to-speech-sts-pipeline-vs-cascaded-architectures)
   - [1.3 Multi-Tenant Isolation & Stateless Process Model](#13-multi-tenant-isolation-stateless-process-model)
   - [1.4 50,000-Token Sliding Window Context Compression](#14-50000-token-sliding-window-context-compression)
   - [1.5 Architecture Diagram](#15-architecture-diagram)
2. [How to Dispatch & Call the Agent](#2-how-to-dispatch-call-the-agent)
   - [2.1 The Job-Dispatch Execution Model](#21-the-job-dispatch-execution-model)
   - [2.2 Dispatch via LiveKit CLI (`lk`)](#22-dispatch-via-livekit-cli-lk)
   - [2.3 Dispatch via Backend Server SDKs (Python, Node.js/TypeScript, Go)](#23-dispatch-via-backend-server-sdks-python-nodejstypescript-go)
   - [2.4 Passing Room Metadata vs. Job Metadata](#24-passing-room-metadata-vs-job-metadata)
   - [2.5 Cascading Metadata Hierarchy & Deep-Merge Semantics](#25-cascading-metadata-hierarchy-deep-merge-semantics)
   - [2.6 Participant Identity Targeting in Multi-Party Rooms](#26-participant-identity-targeting-in-multi-party-rooms)
3. [Complete Metadata Contract Reference (`SessionConfig`)](#3-complete-metadata-contract-reference-sessionconfig)
   - [3.1 Schema Enforcement: Pydantic V2 Fail-Fast Contract](#31-schema-enforcement-pydantic-v2-fail-fast-contract)
   - [3.2 Top-Level Schema Architecture & Hierarchy Overview](#32-top-level-schema-architecture-hierarchy-overview)
   - [3.3 Section 1: `models` (Provider, Model Name, Thinking & Modalities)](#33-section-1-models-provider-model-name-thinking-modalities)
   - [3.4 Section 2: `instructions` (Prompts, Speech Rules & Context Interpolation)](#34-section-2-instructions-prompts-speech-rules-context-interpolation)
   - [3.5 Section 3: `voice` (Voice ID Timbre, Provider, Language & Speed)](#35-section-3-voice-voice-id-timbre-provider-language-speed)
   - [3.6 Section 4: `session` (Watchdogs, Inactivity, Join Timeouts & Cleanup)](#36-section-4-session-watchdogs-inactivity-join-timeouts-cleanup)
   - [3.7 Section 5: `first_turn` (Initiation Modes: say, generate_reply, listen & Greeting Text)](#37-section-5-first_turn-initiation-modes-say-generate_reply-listen-greeting-text)
   - [3.8 Section 6: `turn_detection` & `vad` (Server VAD vs. Silero, Thresholds & Resumption)](#38-section-6-turn_detection-vad-server-vad-vs-silero-thresholds-resumption)
   - [3.9 Section 7: `tools` (Dynamic REST Webhooks, Path Templating & Built-in `end_call`)](#39-section-7-tools-dynamic-rest-webhooks-path-templating-built-in-end_call)
   - [3.10 Section 8: `reporting` (Webhook Telemetry, Structured/Raw Formats & SSRF Security)](#310-section-8-reporting-webhook-telemetry-structuredraw-formats-ssrf-security)
   - [3.11 Section 9: `recording` (Room Audio & Trace Egress Recording)](#311-section-9-recording-room-audio-trace-egress-recording)
   - [3.12 Section 10: `context_compression` (50,000-Token Sliding Window Runtime)](#312-section-10-context_compression-50000-token-sliding-window-runtime)
   - [3.13 Complementary Sections: `agent`, `metadata`, and `version`](#313-complementary-sections-agent-metadata-and-version)
4. [Available Models, Providers & Voice Configuration](#4-available-models-providers-voice-configuration)
   - [4.1 Supported Models & Naming Identifiers](#41-supported-models-naming-identifiers)
   - [4.2 Extended Thinking & Reasoning Budget Configuration](#42-extended-thinking-reasoning-budget-configuration)
   - [4.3 Dynamic Model Promotion Behavior](#43-dynamic-model-promotion-behavior)
   - [4.4 Gemini Live Built-in Voice Catalog](#44-gemini-live-built-in-voice-catalog)
   - [4.5 Language Code Normalization and Locked ASR Codes](#45-language-code-normalization-and-locked-asr-codes)
   - [4.6 Speech Recognition Biasing via Dynamic Custom Vocabulary](#46-speech-recognition-biasing-via-dynamic-custom-vocabulary)
   - [4.7 Extensible Adapter Architecture (`ModelFactory`)](#47-extensible-adapter-architecture-modelfactory)
5. [Dynamic Tool Calling & Webhook Execution](#5-dynamic-tool-calling-webhook-execution)
   - [5.1 Built-in Tool: Graceful Termination (`end_call`) & Audio Playout Draining](#51-built-in-tool-graceful-termination-end_call-audio-playout-draining)
   - [5.2 Dynamic REST Webhook Tools](#52-dynamic-rest-webhook-tools)
   - [5.3 Secret Resolution via `{{env:KEY}}` Syntax](#53-secret-resolution-via-envkey-syntax)
   - [5.4 URL Path Parameter Templating (`{param}`)](#54-url-path-parameter-templating-param)
6. [Post-Session Telemetry & Reporting Dispatcher](#6-post-session-telemetry-reporting-dispatcher)
   - [6.1 Normalized Telemetry Payload Schema: Structured vs. Raw Formats](#61-normalized-telemetry-payload-schema-structured-vs-raw-formats)
   - [6.2 SSRF Protection & Pre-Flight DNS Inspection](#62-ssrf-protection-pre-flight-dns-inspection)
   - [6.3 Retry Engine with Exponential Backoff & Dead-Letter Storage](#63-retry-engine-with-exponential-backoff-dead-letter-storage)
7. [End-to-End JSON Payload Examples](#7-end-to-end-json-payload-examples)
   - [7.1 Minimal Working Configuration](#71-minimal-working-configuration)
   - [7.2 Production Enterprise Full-Featured Configuration](#72-production-enterprise-full-featured-configuration)
   - [7.3 Production Application Blueprint: Tier 1 Customer Support](#73-production-application-blueprint-tier-1-customer-support)
   - [7.4 Production Application Blueprint: Universal Appointment Scheduler](#74-production-application-blueprint-universal-appointment-scheduler)
   - [7.5 Production Application Blueprint: Virtual Receptionist & Inbound Concierge](#75-production-application-blueprint-virtual-receptionist-inbound-concierge)
8. [Error Handling, Safety Guards & Troubleshooting](#8-error-handling-safety-guards-troubleshooting)
   - [8.1 Diagnosing `MetadataValidationError`](#81-diagnosing-metadatavalidationerror)
   - [8.2 SSRF Guard Violations](#82-ssrf-guard-violations)
   - [8.3 WebSocket Timeout, WS 1007 & 300-Second Graceful Call Draining](#83-websocket-timeout-ws-1007-300-second-graceful-call-draining)
   - [8.4 Quick Diagnostic Checklist](#84-quick-diagnostic-checklist)

---

## 1. Executive Overview & Framework Architecture

### 1.1 What is this Agent Framework?

The **LiveKit Voice Agent Framework** is an application-agnostic, low-latency Speech-to-Speech (STS) runtime engine designed for enterprise voice AI services. Unlike monolithic or single-purpose voice bots, this framework contains **zero hardcoded personas, zero domain-specific prompt templates, and zero application assumptions**.

Instead, the framework acts as a **pure execution engine**: 100% of the agent's identity, system instructions, voice timbre, speech recognition biasing, conversational dynamics, tool interfaces, and telemetry webhooks are injected **dynamically at room dispatch time** via LiveKit metadata JSON payloads. A single containerized worker pool can concurrently serve technical support desks, medical appointment coordinators, inbound concierges, or coaching bots without codebase modifications.

### 1.2 Speech-to-Speech (STS) Pipeline vs. Cascaded Architectures

Traditional voice AI bots use a **cascaded pipeline** consisting of three disconnected steps:

```text
[Human Voice] -> (1) STT (Transcription) -> (2) LLM (Text Generation) -> (3) TTS (Audio Synthesis) -> [Agent Voice]
```

Cascaded pipelines introduce three severe engineering bottlenecks:
1. **Compounding Latency**: The user must wait for STT endpointing (~400–700ms), LLM first-token generation (~300–800ms), and TTS audio chunk generation (~400–800ms), producing a sluggish total turn turnaround of **1.5 to 3.0 seconds**.
2. **Loss of Non-Verbal Audio Inflections**: Tone, emotional cadence, laughter, sarcasm, hesitation, and pitch variation are stripped during text transcription.
3. **Severe Interruptions**: Turn-taking is brittle because the TTS synthesizes rigid audio chunks that cannot naturally pause or resume during human backchanneling ("uh-huh", "right").

#### The Native Gemini 3.8 Live Speech-to-Speech Architecture

The LiveKit Voice Agent Framework bypasses transcription cascades by establishing a native, full-duplex WebSocket stream directly to **Google Gemini 3.8 Live** (`livekit-plugins-google`, `google-genai`):

```text
                        +---------------------------------------+
                        |          Caller / Human User          |
                        |   (Browser / SIP Telephony / WebRTC)  |
                        +-------------------+-------------------+
                                            | WebRTC Audio (Opus / PCM)
                                            | 16kHz Audio Frames
                                            v
                        +---------------------------------------+
                        |         LiveKit SFU / Cloud           |
                        |   (Room Metadata & Job Dispatching)   |
                        +-------------------+-------------------+
                                            |
                                            | Outbound WebSocket
                                            v
                        +---------------------------------------+
                        |     LiveKit Agent Framework Worker    |
                        |     (SessionCoordinator on Docker)    |
                        +---------+-------------------+---------+
                                  |                   |
      Direct Audio Token Stream   |                   | Post-Call Telemetry
       (16kHz In / 24kHz Out)     |                   | (SSRF Guarded Webhook)
       Latency: ~500ms - 800ms    |                   |
                                  v                   v
               +-----------------------+   +-----------------------+
               | Google Gemini 3.8 Live|   |   Your Backend API /  |
               |  (or OpenAI Realtime) |   |   Analytics Webhook   |
               +-----------------------+   +-----------------------+
```

- **Sub-Second Latency**: Audio streams directly into Gemini Live in 16kHz PCM chunks and returns in 24kHz PCM chunks. Latency drops to **500–800ms**, indistinguishable from natural human conversation.
- **Acoustic Understanding**: Gemini directly interprets acoustic nuances such as distress, hesitation, speech pacing, and background noise.
- **Bidirectional Interruption**: Audio generation halts immediately when user speech is detected, preventing unnatural talking over callers.

### 1.3 Multi-Tenant Isolation & Stateless Process Model

The framework is strictly multi-tenant safe:
- **No Shared Mutable State**: Configuration, model instances, tool caches, chat history, and telemetry timers are encapsulated strictly within an instance of `SessionContext` tied to the individual LiveKit `JobContext`.
- **Process Isolation**: Each LiveKit room connection runs inside an isolated coroutine lifecycle within the worker process. State cannot bleed across concurrent or sequential calls.
- **Fail-Fast Crash Containment**: A configuration error or unhandled webhook exception in one room tears down that specific room cleanly, executes its post-call reporting hook with error diagnostics, and leaves all sibling sessions unaffected.

### 1.4 50,000-Token Sliding Window Context Compression

In native audio Speech-to-Speech models, audio frames consume tokens continuously at a rate of approximately **25 audio tokens per second** (~1,500 tokens/minute, ~90,000 tokens/hour). 

Without sliding window compression, standard Gemini Live WebSocket sessions exceed default token limits within 15–20 minutes, precipitating fatal **WebSocket Error 1007 (Session Context Exceeded)** closures.

The framework automatically configures Google's native sliding window context compression:
```python
context_window_compression = types.ContextWindowCompressionConfig(
    sliding_window=types.SlidingWindow(
        target_tokens=50000,
    )
)
```
- **How it Works**: When accumulated conversational audio reaches 50,000 tokens, older raw audio chunks are automatically truncated and compressed in the server-side model state, while the active system prompt, context data, and recent conversation turns remain pinned in working memory.
- **Result**: Sessions can run for up to the configured `max_duration_seconds` (e.g., 2 hours) without token saturation or unexpected session termination.

### 1.5 Architecture Diagram

```text
+----------------------------------------------------------------------------------------------------+
|                                      LIVEKIT VOICE AGENT WORKER                                     |
|                                                                                                    |
|  +------------------------+      +--------------------------+      +----------------------------+  |
|  |     LiveKit SFU        |      |      SessionCoordinator   |      |    Cascading Resolver      |  |
|  |  Job Dispatch Handler  | ---> |   (JobContext Lifecycle) | ---> |  (Job + Room + Partic Meta)|  |
|  +------------------------+      +-------------+------------+      +--------------+-------------+  |
|                                                |                                  |                 |
|                                                v                                  v                 |
|                                  +--------------------------+      +----------------------------+  |
|                                  |       ModelFactory       |      |    Pydantic V2 Contract    |  |
|                                  |   (GeminiLiveAdapter)    | <--- |      (SessionConfig)       |  |
|                                  +-------------+------------+      +----------------------------+  |
|                                                |                                                    |
|                   +----------------------------+----------------------------+                       |
|                   |                            |                            |                       |
|                   v                            v                            v                       |
|     +---------------------------+ +--------------------------+ +--------------------------+         |
|     |     LiveKit AgentSession  | |    Dynamic ToolRegistry  | |    ReportingDispatcher   |         |
|     | - TurnHandlingOptions     | | - end_call (Audio Drain) | | - SSRF DNS Validation    |         |
|     | - 50k Sliding Window      | | - WebhookTool (REST API) | | - Exponential Backoff    |         |
|     | - Locked ASR Language     | | - {{env:KEY}} Secrets    | | - Raw / Structured JSON  |         |
|     +---------------------------+ +--------------------------+ +--------------------------+         |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. How to Dispatch & Call the Agent

### 2.1 The Job-Dispatch Execution Model

LiveKit Agents follow a **job-dispatch worker pool topology**:
1. One or more containerized agent workers boot up and establish an outbound persistent control WebSocket to LiveKit SFU (Cloud or Self-Hosted).
2. Workers register their agent capabilities (`agent_name="voice-agent"`) and listen for incoming job dispatches.
3. When your backend or a client application creates a room or submits an explicit job dispatch containing metadata, LiveKit SFU routes the job to an available worker.
4. The worker establishes WebRTC audio tracks with the caller, resolves the metadata, runs the voice interaction, and gracefully cleans up the room upon completion.

### 2.2 Dispatch via LiveKit CLI (`lk`)

The LiveKit CLI (`lk`) allows immediate local testing and dispatching without writing backend integration code.

#### Method 1: Explicit Agent Dispatch into Existing Room (`lk dispatch create`)
When an inbound SIP call or frontend user has already established a room, you dispatch the agent explicitly with job metadata:

```bash
lk dispatch create \
  --room "support-call-001" \
  --agent-name "voice-agent" \
  --metadata "$(cat examples/customer_support/support_metadata.json)"
```

#### Method 2: Create a New Room with Dispatched Agent (`lk dispatch create --new-room`)
To spin up a new unique room and immediately dispatch the voice agent into it in a single command:

```bash
lk dispatch create \
  --new-room \
  --agent-name "voice-agent" \
  --metadata "$(cat examples/appointment_booking/booking_metadata.json)"
```

#### Method 3: Room Creation and Room Metadata Setting (`lk room create` & `lk room update`)
If your architecture provisions rooms before callers arrive, note that `lk room create [options] ROOM_NAME` takes the room name as a positional argument. Room metadata is attached using `lk room update`:

```bash
# 1. Create the room with empty timeout
lk room create support-call-001 --empty-timeout 300

# 2. Attach metadata to the room
lk room update support-call-001 --metadata "$(cat examples/customer_support/support_metadata.json)"
```

When the first caller enters `support-call-001`, LiveKit dispatches an available worker which extracts the room metadata.

---

### 2.3 Dispatch via Backend Server SDKs (Python, Node.js/TypeScript, Go)

Your backend application services can dispatch agents using either **Job Metadata Dispatch** (`create_dispatch`) or **Room Creation Dispatch** (`create_room`).

#### Python (`livekit-api`)

Install package:
```bash
pip install livekit-api
```

```python
import asyncio
import json
from livekit import api

async def dispatch_examples():
    # Initialize Server API Client
    lk_api = api.LiveKitAPI(
        url="wss://your-project.livekit.cloud",
        api_key="your-api-key",
        api_secret="your-api-secret",
    )

    metadata_dict = {
        "version": "1.0",
        "agent": {"name": "billing-support", "role": "customer_support"},
        "models": {
            "provider": "gemini",
            "model": "gemini-3.8-live",
            "temperature": 0.5,
            "thinking": {"enabled": True, "reasoning_effort": "low"},
        },
        "instructions": {
            "system_prompt": "You are Jordan, assisting {customer_name}.",
            "context_data": {"customer_name": "Elena Rostova"},
            "first_turn": {
                "speaker": "agent",
                "mode": "say",
                "text": "Hello Elena, thank you for calling support. How can I help you today?",
            },
        },
        "voice": {"voice_id": "Aoede", "language": "en-US"},
        "tools": [
            {
                "name": "end_call",
                "type": "builtin",
                "config": {"delete_room": True, "drain_timeout_seconds": 3.0},
            }
        ],
        "reporting": {
            "enabled": True,
            "endpoint": {
                "url": "https://api.yourcompany.com/telemetry/calls",
                "headers": {"Authorization": "Bearer {{env:TELEMETRY_API_KEY}}"},
            },
        },
    }
    serialized_meta = json.dumps(metadata_dict)

    # -------------------------------------------------------------
    # Pattern A: Explicit Agent Dispatch (Passes Job Metadata)
    # -------------------------------------------------------------
    dispatch_req = api.CreateAgentDispatchRequest(
        agent_name="voice-agent",
        room="support-call-001",
        metadata=serialized_meta,
    )
    dispatch = await lk_api.agent_dispatch.create_dispatch(dispatch_req)
    print(f"Created Agent Dispatch: ID={dispatch.id} in room={dispatch.room}")

    # -------------------------------------------------------------
    # Pattern B: Room Creation Dispatch (Passes Room Metadata)
    # -------------------------------------------------------------
    room_req = api.CreateRoomRequest(
        name="support-room-002",
        empty_timeout=300,
        metadata=serialized_meta,
    )
    room = await lk_api.room.create_room(room_req)
    print(f"Created Room with Metadata: name={room.name}, sid={room.sid}")

    await lk_api.aclose()

if __name__ == "__main__":
    asyncio.run(dispatch_examples())
```

---

#### Node.js / TypeScript (`livekit-server-sdk`)

Install package:
```bash
npm install livekit-server-sdk
```

```typescript
import { AgentDispatchClient, RoomServiceClient } from 'livekit-server-sdk';
import * as fs from 'fs';

const livekitHost = process.env.LIVEKIT_URL || 'https://your-project.livekit.cloud';
const apiKey = process.env.LIVEKIT_API_KEY || 'your-api-key';
const apiSecret = process.env.LIVEKIT_API_SECRET || 'your-api-secret';

const roomService = new RoomServiceClient(livekitHost, apiKey, apiSecret);
const agentDispatch = new AgentDispatchClient(livekitHost, apiKey, apiSecret);

async function main() {
  const metadata = JSON.parse(fs.readFileSync('./examples/customer_support/support_metadata.json', 'utf-8'));
  metadata.instructions.context_data = { customer_name: 'Marcus Vance' };
  const serialized = JSON.stringify(metadata);

  // -------------------------------------------------------------
  // Pattern A: Explicit Agent Dispatch (Job Metadata)
  // -------------------------------------------------------------
  const dispatch = await agentDispatch.createDispatch('call-usr-998', 'voice-agent', {
    metadata: serialized,
  });
  console.log(`Dispatched agent job: ID=${dispatch.id} to room=call-usr-998`);

  // -------------------------------------------------------------
  // Pattern B: Room Creation Dispatch (Room Metadata)
  // -------------------------------------------------------------
  const room = await roomService.createRoom({
    name: 'call-room-999',
    emptyTimeout: 300,
    metadata: serialized,
  });
  console.log(`Created room with metadata: name=${room.name}, sid=${room.sid}`);
}

main().catch(console.error);
```

---

#### Go (`github.com/livekit/server-sdk-go/v2`)

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/livekit/protocol/livekit"
	lksdk "github.com/livekit/server-sdk-go/v2"
)

func main() {
	host := os.Getenv("LIVEKIT_URL")
	apiKey := os.Getenv("LIVEKIT_API_KEY")
	apiSecret := os.Getenv("LIVEKIT_API_SECRET")

	roomClient := lksdk.NewRoomServiceClient(host, apiKey, apiSecret)
	dispatchClient := lksdk.NewAgentDispatchClient(host, apiKey, apiSecret)

	metadata := map[string]interface{}{
		"version": "1.0",
		"agent": map[string]string{
			"name": "concierge-agent",
			"role": "receptionist",
		},
		"models": map[string]interface{}{
			"provider": "gemini",
			"model":    "gemini-3.8-live",
		},
		"instructions": map[string]interface{}{
			"system_prompt": "You are a warm, professional hotel receptionist.",
			"first_turn": map[string]string{
				"speaker": "agent",
				"mode":    "say",
				"text":    "Welcome to the Grand Hotel. How can I help you?",
			},
		},
		"voice": map[string]string{
			"voice_id": "Puck",
			"language": "en-US",
		},
	}
	metadataBytes, _ := json.Marshal(metadata)

	// -------------------------------------------------------------
	// Pattern A: Explicit Agent Dispatch (Job Metadata)
	// -------------------------------------------------------------
	dispatchReq := &livekit.CreateAgentDispatchRequest{
		AgentName: "voice-agent",
		Room:      "hotel-lobby-call-1",
		Metadata:  string(metadataBytes),
	}
	dispatch, err := dispatchClient.CreateDispatch(context.Background(), dispatchReq)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Created Agent Dispatch: ID=%s in Room=%s\n", dispatch.Id, dispatch.Room)

	// -------------------------------------------------------------
	// Pattern B: Room Creation Dispatch (Room Metadata)
	// -------------------------------------------------------------
	roomReq := &livekit.CreateRoomRequest{
		Name:         "hotel-lobby-call-2",
		EmptyTimeout: 300,
		Metadata:     string(metadataBytes),
	}
	room, err := roomClient.CreateRoom(context.Background(), roomReq)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Created Room %s with SID %s\n", room.Name, room.Sid)
}
```

---

### 2.4 Passing Room Metadata vs. Job Metadata

Understanding how and when to pass **Job Metadata** versus **Room Metadata** is critical for production system design:

| Feature / Dimension | Job Metadata (`JobContext.job.metadata`) | Room Metadata (`JobContext.room.metadata`) |
| :--- | :--- | :--- |
| **Injection Point** | Passed via `lk dispatch create` or `CreateAgentDispatchRequest`. | Passed via `lk room update` or `CreateRoomRequest`. |
| **Scope of Effect** | Strictly bound to that specific agent worker job instance. | Room-wide; accessible to any participant and agent joining the room. |
| **Cascading Priority** | **Layer 1 (Base)**: Forms the starting foundation of the configuration. | **Layer 2 (Intermediate Overlay)**: Deep-merged on top of Job Metadata. |
| **Multi-Agent Rooms** | **Ideal**: Allows dispatching different agents (e.g. Sales bot vs. Compliance listener) to the same room with unique configs. | Shared across all agents entering the room unless overridden by job or participant metadata. |
| **SIP Telephony** | Preferred when a SIP inbound dispatch rule transfers a call to a voice agent. | Preferred when a webhook creates the room ahead of time before telephony dialing completes. |
| **Override Rule** | If the same key exists in both, **Room Metadata overrides Job Metadata**. | Overrides Job Metadata; can be overridden by Remote Participant attributes. |

---

### 2.5 Cascading Metadata Hierarchy & Deep-Merge Semantics

The framework implements a **3-Layer Cascading Metadata Resolver** (`agent_framework.config.resolver.resolve_cascading_metadata`). Configuration merges progressively from broadest scope to most specific caller:

```text
[Layer 1: Job Dispatch Metadata] (ctx.job.metadata)
              |
              v (deep-merged with)
[Layer 2: Room Metadata]        (ctx.room.metadata)
              |
              v (deep-merged with)
[Layer 3: Participant Metadata] (ctx.room.remote_participants[...])
              |
              v
[Strict Pydantic V2 Validation] -> (SessionConfig)
```

#### Merge Semantics:
- **Nested Dictionaries**: Merged recursively. A higher layer can override single nested keys (e.g., overriding `models.temperature` or `instructions.context_data.user_id`) without rewriting the entire parent block.
- **Lists & Scalar Values**: Primitive values (strings, numbers, booleans) and lists (such as `tools` or `custom_vocabulary`) in a higher layer completely replace the lower layer.
- **Participant Attributes**: Any key in a participant's attributes matching a top-level section (e.g., `instructions`, `voice`) merges into that section. Unmapped attributes are automatically routed into `metadata` passthrough telemetry.

---

### 2.6 Participant Identity Targeting in Multi-Party Rooms

In teleconferencing, webinar, or supervision scenarios with multiple human participants in a single LiveKit room:
1. By default, the agent binds to the **first remote participant** who connects.
2. If `session.target_participant_identity` is specified in metadata, the agent ignores other participants and binds audio, transcription, and conversational turn-taking exclusively to the participant matching that exact identity string.
3. If the target participant has not joined when the agent arrives, the agent waits up to `participant_join_timeout_seconds` (default: `30.0`s) before aborting.

---

## 3. Complete Metadata Contract Reference (`SessionConfig`)

### 3.1 Schema Enforcement: Pydantic V2 Fail-Fast Contract

The framework enforces strict validation using Pydantic V2 (`SessionConfig` in `agent_framework.config.schema`):
- **Forbid Extra Keys (`extra="forbid"`)**: Any unrecognized property at root or within any subsection immediately triggers a `MetadataValidationError`. This prevents silent typos (such as `temprature` or `systemprompt`).
- **Zero Backward Compatibility / Zero Guessing**: Required fields have no fallback defaults; missing them halts execution before establishing media tracks.
- **Strict Bounds & Enums**: Numerical parameters (e.g., `temperature`, `max_duration_seconds`) are bounded by explicit min/max limits.

---

### 3.2 Top-Level Schema Architecture & Hierarchy Overview

The table below outlines all sections of the configuration contract, indicating their exact hierarchical location in JSON payloads, model classes, and mandatory status:

| Section Name | JSON Hierarchy Location | Model Class | Status | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`models`** | Top-Level: `"models"` | `ModelsConfig` | **Mandatory** | *None* | LLM provider, model name, hyperparameters, and thinking. |
| **`instructions`** | Top-Level: `"instructions"` | `InstructionsConfig` | **Mandatory** | *None* | System prompt, context interpolation variables, and speech rules. |
| **`voice`** | Top-Level: `"voice"` | `VoiceConfig` | **Mandatory** | *None* | Voice engine, voice ID timbre, speed, and language code. |
| **`session`** | Top-Level: `"session"` | `SessionLifecycleConfig` | Optional | Defaults applied | Watchdogs, call length limits, timeouts, and room deletion. |
| **`first_turn`** | Nested: `"instructions.first_turn"` | `FirstTurnConfig` | Optional | Defaults applied | First-turn greeting and turn initiation mode. |
| **`turn_detection`** | Top-Level: `"turn_detection"` | `TurnDetectionConfig` | Optional | Defaults applied | Conversational dynamics, interruption defense, and endpointing. |
| **`vad`** | Top-Level: `"vad"` | `VADConfig` | Optional | Defaults applied | Voice Activity Detection engine and sensitivity thresholds. |
| **`tools`** | Top-Level: `"tools"` | `list[ToolConfig]` | Optional | `[]` | Dynamic tools (built-in `end_call` and REST webhooks). |
| **`reporting`** | Top-Level: `"reporting"` | `ReportingConfig` | Optional | Disabled | Post-call telemetry webhook, authentication, and SSRF security. |
| **`recording`** | Nested: `"session.recording"` | `RecordingConfig` | Optional | Disabled | LiveKit native session recording for audio and event traces. |
| **`context_compression`**| Runtime-Managed (Adapter) | `ContextWindowCompressionConfig` | **Automatic** | `50000` tokens | Sliding window context compression preventing WS 1007. |
| **`agent`** | Top-Level: `"agent"` | `AgentConfig` | Optional | Defaults applied | Process identity and logging metadata. |
| **`metadata`** | Top-Level: `"metadata"` | `dict[str, Any]` | Optional | `{}` | Arbitrary passthrough key-value pairs preserved in telemetry. |
| **`version`** | Top-Level: `"version"` | `str` | Optional | `"1.0"` | Contract schema specification version. |

> [!CAUTION]
> **Avoid Root-Level Nesting Errors**:
> - `first_turn` **must** be nested inside `"instructions": { "first_turn": { ... } }`. Placing `"first_turn"` at the root payload triggers `MetadataValidationError: Extra inputs are not permitted`.
> - `recording` **must** be nested inside `"session": { "recording": { ... } }`. Placing `"recording"` at the root payload triggers `MetadataValidationError: Extra inputs are not permitted`.

---

### 3.3 Section 1: `models` (Provider, Model Name, Thinking & Modalities)

Governs the AI reasoning core, token budgets, and speech biasing.

```json
"models": {
  "provider": "gemini",
  "model": "gemini-3.8-live",
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 1024,
  "modalities": ["audio"],
  "thinking": {
    "enabled": false,
    "reasoning_effort": "medium",
    "budget_tokens": null,
    "include_thoughts": false
  },
  "custom_vocabulary": ["OAuth2", "Kubernetes", "gRPC", "refund"]
}
```

| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `provider` | `str` | **Mandatory** | *None* | `'gemini'`, `'openai'`, `'pipeline'` | AI model provider engine. |
| `model` | `str` | **Mandatory** | *None* | Non-empty string | Provider model identifier (e.g., `'gemini-3.8-live'`). |
| `temperature` | `float` | Optional | `0.7` | `0.0 <= x <= 2.0` | Sampling randomness. Lower is more deterministic. |
| `top_p` | `float` | Optional | `0.95` | `0.0 <= x <= 1.0` | Nucleus sampling probability threshold. |
| `top_k` | `int` | Optional | `40` | `1 <= x <= 100` | Top-k sampling limit. |
| `max_output_tokens` | `int` | Optional | `1024` | `>= 1` | Maximum generation tokens per single turn. |
| `modalities` | `list[str]` | Optional | `["audio"]` | Subset of `['audio', 'text']` | Active media modalities. Standard STS uses `["audio"]`. |
| `thinking` | `object` | Optional | Defaults applied | See `ThinkingConfig` below | Extended reasoning configuration. |
| `custom_vocabulary` | `list[str]` | Optional | `[]` | List of strings | Words/jargon passed to ASR biasing engine to improve accuracy. |

#### `ThinkingConfig` Object Fields
| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `enabled` | `bool` | Optional | `false` | Boolean | Activates Gemini thinking / reasoning mode before speech. |
| `reasoning_effort` | `str` | Optional | `"medium"` | `'minimal'`, `'low'`, `'medium'`, `'high'` | Thinking depth and deliberation intensity. |
| `budget_tokens` | `int` | Optional | `null` | `>= 0` | Explicit token ceiling (deprecated for Gemini 3.8/3.x; prefer `reasoning_effort`). |
| `include_thoughts` | `bool` | Optional | `false` | Boolean | Whether internal thought traces are surfaced in worker event logs. |

---

### 3.4 Section 2: `instructions` (Prompts, Speech Rules & Context Interpolation)

Controls the system prompt, context variable templating, and spoken dialogue formatting.

```json
"instructions": {
  "system_prompt": "You are Jordan, assisting {customer_name} regarding account {account_id}.",
  "context_data": {
    "customer_name": "Elena Rostova",
    "account_id": "ACC-99214"
  },
  "spoken_dialogue_rules": true,
  "first_turn": {
    "speaker": "agent",
    "mode": "say",
    "text": "Hello Elena, thank you for calling support. How can I help you today?"
  }
}
```

| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `system_prompt` | `str` | **Mandatory** | *None* | `1 <= len <= 65536` | Primary system prompt defining personality and behavior. |
| `context_data` | `dict` | Optional | `{}` | Key-value mapping | Dictionary interpolated into `{key}` placeholders in `system_prompt`. |
| `spoken_dialogue_rules`| `bool` | Optional | `true` | Boolean | Appends universal spoken speech rules (concise turns, no markdown). |
| `first_turn` | `object` | Optional | Defaults applied | See Section 3.7 | Initiation greeting and mode configuration. |

#### Automatic Spoken Dialogue Guidelines Injection
When `spoken_dialogue_rules` is `true`, the framework automatically appends the following prompt guardrail:
```text
Conversational speech guidelines:
- Speak conversationally, naturally, and concisely (typically 1-2 sentences per turn).
- Do not format responses with markdown, bullet points, asterisks, or tables, as this will be spoken aloud.
- Ask clarifying questions when needed, but keep them focused and brief.
```

---

### 3.5 Section 3: `voice` (Voice ID Timbre, Provider, Language & Speed)

Governs acoustic output, synthesized voice persona, speech rate, and language tag.

```json
"voice": {
  "provider": "gemini",
  "voice_id": "Aoede",
  "language": "en-US",
  "speed": 1.05
}
```

| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `provider` | `str \| null`| Optional | Inherits `models.provider` | `'gemini'`, `'openai'`, `'elevenlabs'`, `'pipeline'` | Voice synthesis engine. |
| `voice_id` | `str` | **Mandatory** | *None* | Non-empty string | Provider voice identifier (e.g., `'Aoede'`, `'Puck'`, `'Charon'`). |
| `language` | `str` | Optional | `"en-US"` | BCP-47 language tag | Normalized BCP-47 tag. Locks ASR transcription language. |
| `speed` | `float` | Optional | `1.0` | `0.0 < x <= 4.0` | Playback speed multiplier (e.g., `1.05` for brisk pacing). |

---

### 3.6 Section 4: `session` (Watchdogs, Inactivity, Join Timeouts & Cleanup)

Manages safety watchdogs, room deletion policies, and participant binding.

```json
"session": {
  "max_duration_seconds": 1800,
  "inactivity_timeout_seconds": 120,
  "participant_join_timeout_seconds": 30.0,
  "delete_room_on_terminate": true,
  "target_participant_identity": null,
  "recording": {
    "enabled": false,
    "audio": true,
    "traces": true
  }
}
```

| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `max_duration_seconds` | `int` | Optional | `1800` (30 min) | `30 <= x <= 7200` (2 hrs) | Hard cap on total session duration. Terminated by watchdog. |
| `inactivity_timeout_seconds` | `int` | Optional | `120` (2 min) | `1 <= x <= 3600` | Disconnects room if no voice interactions occur. |
| `participant_join_timeout_seconds`| `float` | Optional | `30.0` | `0.0 < x <= 600.0` | Maximum wait for participant before aborting worker job. |
| `delete_room_on_terminate` | `bool` | Optional | `true` | Boolean | Closes and purges LiveKit room via API upon disconnect. |
| `target_participant_identity` | `str \| null`| Optional | `null` | String or null | Specific participant identity to bind with in multi-party rooms. |
| `recording` | `object` | Optional | Defaults applied | See Section 3.11 | Native LiveKit session recording parameters. |

---

### 3.7 Section 5: `first_turn` (Initiation Modes: say, generate_reply, listen & Greeting Text)

Defines conversational initiation behavior upon room connection. In the metadata payload, this configuration is nested under `instructions.first_turn`:

```json
"instructions": {
  "system_prompt": "You are a customer support agent.",
  "first_turn": {
    "speaker": "agent",
    "mode": "say",
    "text": "Hello, thank you for calling support. How can I help you today?"
  }
}
```

| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `speaker` | `str` | Optional | `"user"` | `'agent'` or `'user'` | Specifies who initiates the first turn. |
| `mode` | `str` | Optional | `"listen"` | `'say'`, `'generate_reply'`, `'listen'` | Agent action upon entering the room. |
| `text` | `str \| null`| Conditional | `null` | Required if `mode='say'` | Exact phrase spoken immediately over WebRTC upon connection. |

#### Operational Modes:
- **`say`**: The agent speaks the exact literal text provided in `text` immediately over WebRTC (supported on pipeline models with TTS or models with `supports_say: true`).
- **`generate_reply`**: The agent queries the model to generate a contextual greeting based on the system prompt and context variables (supported on pipeline models with TTS).
- **`listen`**: The agent joins silently and waits for the caller to speak first.

> [!IMPORTANT]
> **Speech-to-Speech (STS) Models & First Message Behavior**:
> - Native STS models like Google Gemini Live (`gemini-3.8-live` and `gemini-3.8-live-extended-thinking`) operate natively audio-to-audio without an attached Text-to-Speech (TTS) engine (`supports_say=False`).
> - Direct `session.say()` calls fail on STS models without TTS, and synthetic `generate_reply()` calls trigger protocol timeouts (such as the 5-second `generation_created` timeout in `livekit-plugins-google`) or event loop blocks while awaiting audio playout.
> - Per LiveKit and Gemini Live best practices, initial agent greeting messages are automatically removed/skipped on STS models lacking `say`/TTS support. The agent connects silently in listening mode, allowing natural, low-latency bidirectional voice dialogue as soon as the user speaks.

#### Validation Rules:
1. If `mode` is `"say"`, `text` **must** be provided and non-empty. `speaker` automatically normalizes to `"agent"`.
2. If `speaker` is `"agent"` and `mode` is not specified, `mode` defaults to `"say"` (if `text` is provided) or `"generate_reply"`.
3. An agent cannot have `speaker="agent"` with `mode="listen"` (raises validation error).
4. A user cannot have `speaker="user"` with `mode="say"` (raises validation error).

---

### 3.8 Section 6: `turn_detection` & `vad` (Server VAD vs. Silero, Thresholds & Resumption)

The framework decomposes speech dynamics into two dedicated top-level configurations:
1. **`vad` (`VADConfig`)**: Physical audio frame voice onset detection.
2. **`turn_detection` (`TurnDetectionConfig`)**: Conversational turn-taking, endpointing, and interruption handling.

#### A. Voice Activity Detection (`vad`)
```json
"vad": {
  "provider": "native",
  "disabled": false,
  "silence_duration_ms": 1500,
  "prefix_padding_ms": 300,
  "threshold": 0.5
}
```

| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `provider` | `str` | Optional | `"native"` | `'native'` or `'silero'` | VAD engine (`native` LiveKit server VAD or local `silero`). |
| `disabled` | `bool` | Optional | `false` | Boolean | If `true`, disables automated voice onset detection. |
| `silence_duration_ms` | `int` | Optional | `1500` | `200 <= x <= 3000` | Milliseconds of silence after speech before endpointing. |
| `prefix_padding_ms` | `int` | Optional | `300` | `0 <= x <= 2000` | Pre-speech audio buffer retained to prevent clipping initial syllables. |
| `threshold` | `float` | Optional | `0.5` | `0.0 <= x <= 1.0` | Speech detection probability threshold. |

#### B. Turn Detection & Interruption Dynamics (`turn_detection`)
```json
"turn_detection": {
  "mode": "server_vad",
  "allow_interruptions": true,
  "min_interruption_duration_ms": 500,
  "min_interruption_words": 2,
  "resume_false_interruption": true,
  "min_endpointing_delay_ms": 500,
  "user_away_timeout_seconds": 20.0
}
```

| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mode` | `str` | Optional | `"server_vad"` | String | Turn detection mode (`"server_vad"`, `"realtime_llm"`, `"stt"`). |
| `allow_interruptions` | `bool` | Optional | `true` | Boolean | Whether human speech interrupts active agent audio playout. |
| `min_interruption_duration_ms` | `int` | Optional | `500` | `100 <= x <= 2000` | Continuous speech required before agent stops speaking. |
| `min_interruption_words` | `int` | Optional | `2` | `>= 0` | Word count threshold required to trigger interruption. |
| `resume_false_interruption` | `bool` | Optional | `true` | Boolean | Resumes agent speech if human input was brief backchanneling ("yeah", "uh-huh"). |
| `min_endpointing_delay_ms` | `int` | Optional | `500` | `0 <= x <= 5000` | Minimum silence delay before finalizing turn. |
| `user_away_timeout_seconds` | `float` | Optional | `20.0` | `0.0 < x <= 300.0` | Silence duration before triggering user-away handling. |

> **Nested Configuration Compatibility**: The `turn_detection` validator automatically unpacks legacy nested blocks such as `interruption: {"enabled": true, "min_words": 2}` and `endpointing: {"min_delay_ms": 500}` into their flat equivalents.

---

### 3.9 Section 7: `tools` (Dynamic REST Webhooks, Path Templating & Built-in `end_call`)

Defines dynamic functions exposed to the model. Configured as an array of `ToolConfig` items.

```json
"tools": [
  {
    "name": "lookup_order",
    "type": "webhook",
    "description": "Look up tracking status and order items by Order ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {"type": "string", "description": "e.g. ORD-1092"}
      },
      "required": ["order_id"]
    },
    "config": {
      "url": "https://api.example.com/v1/orders/{order_id}",
      "method": "GET",
      "headers": {
        "Authorization": "Bearer {{env:COMMERCE_API_KEY}}"
      },
      "timeout_seconds": 5.0
    }
  },
  {
    "name": "end_call",
    "type": "builtin",
    "description": "Gracefully terminates the call when finished.",
    "config": {
      "delete_room": true,
      "drain_timeout_seconds": 3.0,
      "farewell_instruction": "Thank the caller warmly before hanging up."
    }
  }
]
```

#### Common `ToolConfig` Fields
| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `name` | `str` | **Mandatory** | *None* | Must be unique in `tools` | Identifier exposed to LLM function calling schema. |
| `type` | `str` | Optional | `"webhook"` | `'builtin'` or `'webhook'` | Tool architecture type. |
| `description` | `str` | Optional | `""` | String | Detailed instructions for the LLM explaining when to call the tool. |
| `parameters` | `dict` | Optional | `{}` | Standard JSON Schema | JSON Schema object defining required arguments and types. |
| `config` | `dict` | Optional | `{}` | Key-value dictionary | Configuration specific to the tool type. |

#### Built-in Tool: `end_call`
Supported config parameters:
- `delete_room` (`bool`, default: `true`): Deletes the LiveKit room via Server API.
- `drain_timeout_seconds` (`float`, default: `5.0`): Maximum seconds allocated to let the model's final farewell audio play completely over WebRTC before closing tracks.
- `farewell_instruction` (`str`): Prompt instruction guiding the final farewell response.

#### Dynamic Webhook Tools (`type="webhook"`)
Supported config parameters:
- `url` (`str`, **Mandatory**): HTTP(S) endpoint URL. Supports `{param}` path placeholders.
- `method` (`str`, default: `"POST"`): HTTP method (`"GET"`, `"POST"`, `"PUT"`).
- `headers` (`dict[str, str]`, default: `{}`): Custom headers. Supports `{{env:KEY}}` references.
- `timeout_seconds` (`float`, default: `5.0`): Network timeout in seconds.

---

### 3.10 Section 8: `reporting` (Webhook Telemetry, Structured/Raw Formats & SSRF Security)

Dispatches comprehensive session analytics, chat logs, token usage, and timeline events to your API upon call completion.

```json
"reporting": {
  "enabled": true,
  "endpoint": {
    "url": "https://api.yourcompany.com/v1/telemetry",
    "method": "POST",
    "headers": {
      "Authorization": "Bearer {{env:ANALYTICS_KEY}}"
    },
    "timeout_seconds": 10.0
  },
  "auth": {
    "type": "bearer",
    "secret_ref": "env:ANALYTICS_TOKEN"
  },
  "retry_policy": {
    "max_retries": 3,
    "initial_delay_ms": 1000,
    "backoff_factor": 2.0
  },
  "security": {
    "validate_ssrf": true,
    "allow_local": false,
    "allowed_domains": ["api.yourcompany.com"]
  },
  "payload_config": {
    "include_chat_history": true,
    "include_model_usage": true,
    "include_events": true,
    "custom_fields": {
      "tenant_id": "tenant-corp-01"
    }
  }
}
```

| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `enabled` | `bool` | Optional | Auto-detected | Defaults to `true` if `endpoint.url` given | Enables post-call telemetry dispatching. |
| `endpoint` | `object \| null`| Conditional | `null` | Required if `enabled=true` | Target webhook URL and connection options. |
| `auth` | `object \| null`| Optional | `null` | Requires both `type` and `secret_ref` | Webhook authentication configuration. |
| `retry_policy` | `object` | Optional | Defaults applied | See below | Exponential backoff retry engine. |
| `security` | `object` | Optional | Defaults applied | See below | SSRF protection and IP blocking rules. |
| `payload_config` | `object` | Optional | Defaults applied | See below | Telemetry format and inclusion switches. |

#### Sub-Configurations:
- **`endpoint`**: `url` (str, required), `method` (`"POST"` | `"PUT"`), `headers` (dict with `{{env:KEY}}` support), `timeout_seconds` (0.0 < x <= 120.0).
- **`auth`**: `type` (`"bearer"`, `"api_key"`, `"hmac_sha256"`), `secret_ref` (`"env:VAR_NAME"`).
- **`retry_policy`**: `max_retries` (0–10, default 3), `initial_delay_ms` (0–60000, default 1000), `backoff_factor` (1.0–10.0, default 2.0).
- **`security`**: `validate_ssrf` (bool, default `true`), `allow_local` (bool, default `false`), `allowed_domains` (optional list of domains).
- **`payload_config`**: `include_chat_history` (bool, default `true`), `include_model_usage` (bool, default `true`), `include_events` (bool, default `true`), `custom_fields` (dict).

---

### 3.11 Section 9: `recording` (Room Audio & Trace Egress Recording)

Controls LiveKit native server-side session recording and event tracing. In the metadata payload, this is nested under `session.recording`:

```json
"session": {
  "recording": {
    "enabled": false,
    "audio": true,
    "traces": true
  }
}
```

| Field | Type | Mandatory? | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `enabled` | `bool` | Optional | `false` | Enables LiveKit native session recording. |
| `audio` | `bool` | Optional | `true` | Records WebRTC audio track to storage bucket. |
| `traces` | `bool` | Optional | `true` | Records execution timeline and diagnostic trace events. |

> [!WARNING]
> Do not declare `"recording"` as a root-level key in your JSON payload. It is defined within `SessionLifecycleConfig` and must be placed inside `"session": { "recording": { ... } }`.

---

### 3.12 Section 10: `context_compression` (50,000-Token Sliding Window Runtime)

Full-duplex Speech-to-Speech audio consumes tokens continuously at **~25 audio tokens per second** (~1,500 tokens/minute). Without automated context compression:
- Audio sessions naturally exceed token limits within 15 to 20 minutes.
- The WebSocket connection terminates abnormally with **LiveKit / Gemini Error 1007: Context Window Exceeded**.

#### Configuration Specification:
- **Architecture Level**: Configured at the model adapter level (`GeminiLiveAdapter` in `src/agent_framework/models/gemini_live.py`).
- **Implementation**: Google GenAI `types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow(target_tokens=50000))`.
- **Target Tokens**: `50,000` sliding window tokens.
- **Operational Behavior**: As conversational turns proceed, older audio chunks are smoothly rolled out of working token memory, while the initial system instructions, dynamic context data, and recent conversation turns are pinned and retained.
- **Developer Impact**: Zero manual token truncation or memory management is required in application payloads.

---

### 3.13 Complementary Sections: `agent`, `metadata`, and `version`

#### A. `agent` (`AgentConfig`)
Encapsulates worker process identity used for logging, metrics labeling, and telemetry reports.

```json
"agent": {
  "name": "tier1-billing-agent",
  "role": "technical_support",
  "description": "Customer Billing and Tier 1 Triage Agent"
}
```

| Field | Type | Mandatory? | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `name` | `str` | Optional | `"voice-agent"` | Non-empty, non-whitespace | Logical name for worker logs and telemetry traces. |
| `role` | `str` | Optional | `"assistant"` | Non-empty, non-whitespace | Functional role (e.g., `assistant`, `receptionist`, `scheduler`). |
| `description`| `str \| null`| Optional | `null` | String or null | High-level summary of agent intent. |

#### B. `metadata` (`dict[str, Any]`)
A top-level dictionary (`default: {}`) for passing application-specific telemetry, customer IDs, SIP telephony header details, or routing tags. This dictionary is echoed back directly in post-session reporting webhooks under `payload["metadata"]`.

#### C. `version` (`str`)
Contract schema specification version (default: `"1.0"`).

---

## 4. Available Models, Providers & Voice Configuration

### 4.1 Supported Models & Naming Identifiers

The framework integrates natively with Google Gemini Live via `livekit-plugins-google` and `google-genai`:

| Model Identifier | Provider | Modal Options | Latency | Recommended Use Cases |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3.8-live` | `gemini` | `["audio"]`, `["audio", "text"]` | ~500ms | **Default production model**. Fast conversational STS, low latency, natural acoustics. |
| `gemini-3.8-live-extended-thinking` | `gemini` | `["audio"]`, `["audio", "text"]` | ~700ms | Complex multi-step reasoning, medical triage, legal/billing troubleshooting. |
| `gemini-2.0-flash-exp` | `gemini` | `["audio"]`, `["audio", "text"]` | ~600ms | Experimental fallback / compatibility testing. |

### 4.2 Extended Thinking & Reasoning Budget Configuration

Gemini 3.8 Live supports real-time reasoning prior to synthesizing audio via `gemini-3.8-live-extended-thinking`. To enable thinking, configure `models.thinking`:

```json
"thinking": {
  "enabled": true,
  "reasoning_effort": "low",
  "include_thoughts": false
}
```

- **`reasoning_effort` options**:
  - `"minimal"`: Auto-normalized to `"low"` on Gemini 3.8 Live Extended Thinking (backend requires `"low"`, `"medium"`, or `"high"`).
  - `"low"`: Balanced reasoning for structured tool picking (~200–400ms latency impact).
  - `"medium"`: Deeper analysis for technical support (~400–700ms latency impact).
  - `"high"`: Exhaustive deliberation for complex diagnoses (~800–1200ms latency impact).
- **`budget_tokens` (Deprecated in Gemini 3.8/3.x)**:
  - Google has deprecated the numeric `budget_tokens` parameter across Gemini 3.x in favor of the `reasoning_effort` enum (`thinking_level`).
  - **Protocol Mutual Exclusivity**: The Gemini Live WebSocket protocol strictly rejects having both `thinking_budget` and `thinking_level` set simultaneously (causing WebSocket close error 1007: `google.genai.errors.APIError: 1007 None. You can only set only one of thinking budget and thinking level.`).
  - In `GeminiLiveAdapter`, `thinking_level` is authoritative and `thinking_budget` is set to `None` for all Gemini 3.8 models, ensuring clean WebSocket handshakes without protocol rejection.

### 4.3 Dynamic Model Promotion Behavior

The Google Gemini API disallows sending `thinking_level` or thinking parameters to the standard `gemini-3.8-live` model endpoint. To provide seamless developer experience:
- If a developer specifies `model: "gemini-3.8-live"` **and** sets `thinking.enabled: true`, the framework's `GeminiLiveAdapter` **automatically promotes the model** to `gemini-3.8-live-extended-thinking`.
- The promotion is logged in worker diagnostics without crashing the session.

### 4.4 Gemini Live Built-in Voice Catalog

Gemini Live provides a rich set of built-in voice timbres. Set via `voice.voice_id`:

| Voice ID | Timbre & Characteristics | Pitch / Style | Recommended Use Cases |
| :--- | :--- | :--- | :--- |
| **`Aoede`** | Warm, composed, professional, empathetic | Natural female timbre | Customer support, healthcare, appointment booking |
| **`Puck`** | Crisp, energetic, engaging, friendly | Expressive male timbre | Concierge, appointment booking, sales reception |
| **`Charon`** | Deep, authoritative, measured, serious | Deep male timbre | Security, legal assistance, formal financial inquiries |
| **`Fenrir`** | Dynamic, clear, articulate, confident | Medium-low male timbre | Technical troubleshooting, IT helpdesk |
| **`Kore`** | Gentle, soothing, attentive, pleasant | Soft female timbre | Mental wellness, hotel concierge, patient care |

### 4.5 Language Code Normalization and Locked ASR Codes

To prevent the agent from accidentally drifting across languages when hearing non-English background speech or loanwords, the framework locks ASR language codes:
1. `voice.language` takes a standard BCP-47 tag (e.g., `"en-US"`, `"es-ES"`, `"fr-FR"`, `"de-DE"`, `"hi-IN"`).
2. The framework normalizes short codes (e.g., `"en"` -> `"en-US"`) via `agent_framework.config.settings.resolve_bcp47_language`.
3. The locked code is passed directly to Gemini Live's `AudioTranscriptionConfig(language_codes=[...])` for both input ASR and output synthesis, locking the recognition engine to the target locale.

### 4.6 Speech Recognition Biasing via Dynamic Custom Vocabulary

STS engines can mishear rare brand names, technical terms, or medical jargon. Pass specific terminology in `models.custom_vocabulary`:

```json
"custom_vocabulary": [
  "OAuth2",
  "Kubernetes",
  "PostgreSQL",
  "triage",
  "Palo Alto Networks"
]
```
These tokens are injected into Gemini's `AudioTranscriptionConfig.custom_vocabulary`, significantly boosting recognition accuracy for specialized vocabulary.

### 4.7 Extensible Adapter Architecture (`ModelFactory`)

The framework decouples model initialization through the `ModelAdapter` abstract base class and `ModelFactory` registry:
- **Default Provider**: `gemini` (`GeminiLiveAdapter`).
- **Pluggable Registration**: Additional providers (such as OpenAI Realtime or custom self-hosted pipelines) can be registered at runtime without modifying core coordinator code:

```python
from agent_framework.models.factory import ModelFactory
from my_custom_package import CustomRealtimeAdapter

ModelFactory.register_adapter("openai", CustomRealtimeAdapter)
```

---

## 5. Dynamic Tool Calling & Webhook Execution

### 5.1 Built-in Tool: Graceful Termination (`end_call`) & Audio Playout Draining

Voice agents must conclude calls gracefully. If an agent cuts the WebRTC connection abruptly while saying "Goodbye, have a great day!", the user only hears "Goodb—" before silence.

The built-in `end_call` tool implements **Utterance-Drain Synchronization**:
1. When the model invokes `end_call`, the tool sets an internal completion state.
2. The agent speaks its final farewell utterance.
3. The tool attaches a listener to the audio output queue and waits up to `drain_timeout_seconds` (default: `5.0`s) until the audio track buffer has completely drained across WebRTC to the user's speaker.
4. Only after the audio buffer drains does the framework close the WebRTC room tracks and invoke LiveKit API `DeleteRoom`.

```json
{
  "name": "end_call",
  "type": "builtin",
  "description": "Gracefully terminates the phone call when customer inquiry is concluded.",
  "config": {
    "delete_room": true,
    "drain_timeout_seconds": 3.0,
    "farewell_instruction": "Thank the caller for their time and wish them a wonderful day before hanging up."
  }
}
```

### 5.2 Dynamic REST Webhook Tools

You can declare arbitrary REST API tools in metadata. The framework compiles these into native LiveKit `FunctionTool` definitions on the fly:

```json
{
  "name": "check_flight_status",
  "type": "webhook",
  "description": "Retrieves real-time departure and arrival status for a commercial flight.",
  "parameters": {
    "type": "object",
    "properties": {
      "flight_number": {
        "type": "string",
        "description": "Flight code, e.g. AA104 or DL492"
      }
    },
    "required": ["flight_number"]
  },
  "config": {
    "url": "https://api.aviation.example.com/v1/flights/{flight_number}",
    "method": "GET",
    "headers": {
      "Authorization": "Bearer {{env:FLIGHT_API_KEY}}",
      "Accept": "application/json"
    },
    "timeout_seconds": 4.0
  }
}
```

### 5.3 Secret Resolution via `{{env:KEY}}` Syntax

To prevent sensitive credentials (API keys, bearer tokens) from being exposed in metadata payloads traversing client applications or networks, the framework supports the `{{env:KEY}}` syntax:
```json
"headers": {
  "Authorization": "Bearer {{env:PAYMENT_SERVICE_KEY}}",
  "X-Tenant-Secret": "{{env:TENANT_SECRET}}"
}
```
- At session boot time, `agent_framework.utils.secrets.resolve_secrets` scans all strings in headers, URLs, and auth blocks.
- References matching `{{env:KEY}}` are looked up directly from host environment variables.
- If an environment variable is missing, a structured `SecretResolutionError` is raised immediately before establishing connections.

### 5.4 URL Path Parameter Templating (`{param}`)

When executing a webhook tool, arguments passed by the model automatically populate matching `{param}` placeholders in the URL:
- Tool parameter: `{"order_id": "99401"}`
- Webhook URL: `"https://api.example.com/v1/orders/{order_id}/tracking"`
- Effective URL: `"https://api.example.com/v1/orders/99401/tracking"`
- Arguments not consumed by URL path parameters are automatically formatted as JSON in the HTTP request body (for `POST` and `PUT` requests) or as query string parameters (for `GET` requests).

---

## 6. Post-Session Telemetry & Reporting Dispatcher

### 6.1 Normalized Telemetry Payload Schema: Structured vs. Raw Formats

Upon call conclusion (whether normal disconnect, `end_call` invocation, timeout, or error), `ReportingDispatcher` sends an HTTP POST/PUT payload to the configured reporting URL.

#### Telemetry Format Modes:
- **Structured JSON Telemetry (`format: structured`)**: The default format generated by `build_session_telemetry_payload`. Turns are normalized into typed roles (`user`, `assistant`, `tool`), model usage metrics are aggregated, and timeline events are sequenced.
- **Raw Telemetry Elements (`format: raw`)**: Preserves raw LiveKit `ChatContext` objects, un-flattened diagnostic traces, and un-parsed caller metadata pass-through.

#### Standardized Payload Structure:
```json
{
  "session_id": "ses_01JH882A9BZ",
  "job_id": "AJ_89b21f00",
  "room_name": "support-call-001",
  "agent": {
    "name": "tier1-billing-agent",
    "role": "technical_support"
  },
  "timestamps": {
    "started_at": "2026-11-21T14:30:00.124Z",
    "ended_at": "2026-11-21T14:35:12.802Z",
    "duration_seconds": 312.678
  },
  "termination": {
    "reason": "completed",
    "error": null
  },
  "chat_history": [
    {
      "role": "assistant",
      "content": "Thank you for calling support. How can I help you today?",
      "timestamp": "2026-11-21T14:30:02.100Z"
    },
    {
      "role": "user",
      "content": "Hi, I have a question about my invoice.",
      "timestamp": "2026-11-21T14:30:06.400Z"
    },
    {
      "role": "assistant",
      "content": "Tool call: lookup_account({"customer_id": "ACC-101"})",
      "timestamp": "2026-11-21T14:30:08.200Z"
    },
    {
      "role": "tool",
      "content": "{"status": "active", "balance": 0.0}",
      "timestamp": "2026-11-21T14:30:09.100Z"
    }
  ],
  "model_usage": [
    {
      "prompt_tokens": 1420,
      "completion_tokens": 850,
      "total_tokens": 2270
    }
  ],
  "events": [
    {"type": "participant_joined", "timestamp": "2026-11-21T14:30:01.000Z"},
    {"type": "first_turn_started", "timestamp": "2026-11-21T14:30:02.050Z"},
    {"type": "tool_executed", "name": "lookup_account", "duration_ms": 340}
  ],
  "custom_fields": {
    "tenant_id": "tenant-corp-01",
    "priority": "high"
  },
  "metadata": {
    "channel": "inbound_sip",
    "queue": "billing"
  }
}
```

### 6.2 SSRF Protection & Pre-Flight DNS Inspection

To prevent attackers from using metadata webhook declarations to probe internal infrastructure or extract cloud credentials, all outbound webhooks (`tools` and `reporting`) run through `agent_framework.reporting.security.is_safe_webhook_url`:

#### Blocked IP Ranges & Hostnames:
1. **Cloud Metadata Endpoints (Strictly Blocked)**:
   - AWS / GCP / Azure IMDS: `169.254.169.254`
   - Google Internal Metadata: `metadata.google.internal`
   - Azure Internal Metadata: `metadata.azure.internal`
   - AWS IPv6 IMDS: `fd00:ec2::254`
   - Alibaba Cloud Metadata: `100.100.100.200`
2. **Loopback & Local Addresses**: `127.0.0.0/8`, `::1`, `localhost` (unless `allow_local: true` for testing).
3. **RFC 1918 Private Networks**: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`.
4. **Link-Local & Carrier-Grade NAT**: `169.254.0.0/16`, `100.64.0.0/10`.
5. **DNS Rebinding Defense**: Performs pre-dispatch DNS resolution via `socket.getaddrinfo` and inspects **every resolved IP address**. If any resolved IP falls in a forbidden subnet, the request is rejected immediately with an `SSRFSecurityError`.

### 6.3 Retry Engine with Exponential Backoff & Dead-Letter Storage

Telemetry dispatch executes asynchronously with exponential backoff:
- Default: 3 retry attempts.
- Initial delay: 1,000ms.
- Backoff multiplier: 2.0 (1s -> 2s -> 4s) with jitter.
- Fail-fast: Non-retryable 4xx client errors (400, 401, 403, 404) fail immediately without wasteful retries.
- Dead-Letter Storage: If all delivery attempts fail, the full payload is safely saved as a JSON dead-letter record under `logs/dead_letter/` to prevent telemetry loss.
- Non-blocking: Telemetry delivery runs as an independent task during teardown so it never delays room freeing.

---

## 7. End-to-End JSON Payload Examples

### 7.1 Minimal Working Configuration

The simplest valid JSON payload containing only mandatory fields:

```json
{
  "version": "1.0",
  "models": {
    "provider": "gemini",
    "model": "gemini-3.8-live"
  },
  "instructions": {
    "system_prompt": "You are a friendly, helpful conversational AI assistant. Keep all responses brief, natural, and conversational."
  },
  "voice": {
    "voice_id": "Aoede"
  }
}
```

---

### 7.2 Production Enterprise Full-Featured Configuration

Demonstrating all schema sections configured for a high-volume enterprise deployment:

```json
{
  "version": "1.0",
  "agent": {
    "name": "enterprise-concierge",
    "role": "concierge",
    "description": "High-priority corporate concierge voice assistant"
  },
  "models": {
    "provider": "gemini",
    "model": "gemini-3.8-live",
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    "modalities": ["audio"],
    "thinking": {
      "enabled": true,
      "reasoning_effort": "low",
      "include_thoughts": false
    },
    "custom_vocabulary": [
      "Executive Briefing Center",
      "Boardroom Alpha",
      "Polycom",
      "catering"
    ]
  },
  "instructions": {
    "system_prompt": "You are Morgan, the executive concierge for {organization}. Assist VIP guests with conference room scheduling, guest badges, and building directions. Always be courteous, precise, and concise.",
    "context_data": {
      "organization": "Acme Global Headquarters"
    },
    "spoken_dialogue_rules": true,
    "first_turn": {
      "speaker": "agent",
      "mode": "say",
      "text": "Good day! Welcome to Acme Global. My name is Morgan. How may I direct your visit today?"
    }
  },
  "voice": {
    "provider": "gemini",
    "voice_id": "Aoede",
    "language": "en-US",
    "speed": 1.05
  },
  "vad": {
    "provider": "native",
    "disabled": false,
    "silence_duration_ms": 1400,
    "prefix_padding_ms": 300,
    "threshold": 0.5
  },
  "turn_detection": {
    "mode": "server_vad",
    "allow_interruptions": true,
    "min_interruption_duration_ms": 450,
    "min_interruption_words": 2,
    "resume_false_interruption": true,
    "min_endpointing_delay_ms": 500,
    "user_away_timeout_seconds": 25.0
  },
  "session": {
    "max_duration_seconds": 1800,
    "inactivity_timeout_seconds": 120,
    "participant_join_timeout_seconds": 30.0,
    "delete_room_on_terminate": true,
    "recording": {
      "enabled": false,
      "audio": true,
      "traces": true
    }
  },
  "tools": [
    {
      "name": "lookup_room_availability",
      "type": "webhook",
      "description": "Check real-time availability of conference rooms.",
      "parameters": {
        "type": "object",
        "properties": {
          "floor": {"type": "integer", "description": "Building floor number"},
          "capacity": {"type": "integer", "description": "Required seat count"}
        },
        "required": ["floor"]
      },
      "config": {
        "url": "https://api.acme.example.com/v1/facilities/rooms",
        "method": "POST",
        "headers": {
          "Authorization": "Bearer {{env:FACILITIES_API_KEY}}",
          "Content-Type": "application/json"
        },
        "timeout_seconds": 5.0
      }
    },
    {
      "name": "end_call",
      "type": "builtin",
      "description": "Conclude the conversation when guest is assisted.",
      "config": {
        "delete_room": true,
        "drain_timeout_seconds": 3.0,
        "farewell_instruction": "Wish the guest a productive meeting and pleasant stay."
      }
    }
  ],
  "reporting": {
    "enabled": true,
    "endpoint": {
      "url": "https://telemetry.acme.example.com/v1/voice-events",
      "method": "POST",
      "headers": {
        "Authorization": "Bearer {{env:TELEMETRY_KEY}}"
      },
      "timeout_seconds": 10.0
    },
    "retry_policy": {
      "max_retries": 3,
      "initial_delay_ms": 1000,
      "backoff_factor": 2.0
    },
    "security": {
      "validate_ssrf": true,
      "allow_local": false
    },
    "payload_config": {
      "include_chat_history": true,
      "include_model_usage": true,
      "include_events": true,
      "custom_fields": {
        "facility_code": "HQ-WEST",
        "tier": "vip"
      }
    }
  },
  "metadata": {
    "kiosk_id": "KIOSK-LOBBY-01"
  }
}
```

---

### 7.3 Production Application Blueprint: Tier 1 Customer Support

Available in repository at: `examples/customer_support/support_metadata.json`

- **Scenario**: Technical troubleshooting and billing inquiries for a cloud hosting provider.
- **Key Features**: Thinking model enabled for troubleshooting steps, REST webhook to query CRM account state, speech biasing for tech terms, and audio drain conclusion.

```json
{
  "version": "1.0",
  "agent": {
    "name": "support-tier1-agent",
    "role": "technical_support",
    "description": "Tier 1 Technical and Billing Customer Support Voice Agent"
  },
  "models": {
    "provider": "gemini",
    "model": "gemini-3.8-live",
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    "modalities": ["audio"],
    "thinking": {
      "enabled": true,
      "reasoning_effort": "low",
      "include_thoughts": false
    },
    "custom_vocabulary": [
      "OAuth",
      "webhook",
      "API key",
      "two-factor authentication",
      "invoice",
      "subscription tier",
      "SSO"
    ]
  },
  "instructions": {
    "system_prompt": "You are Jordan, a helpful and technical Tier 1 Customer Support specialist for CloudServices. Your goal is to troubleshoot account connectivity issues, clarify billing questions, and assist users with service configuration. Always remain patient, clear, and focused on resolution. If the caller needs to update payment details, direct them to use the secure self-service portal at {portal_url}. Conclude the call pleasantly once their issue is resolved.",
    "first_turn": {
      "speaker": "agent",
      "mode": "say",
      "text": "Thank you for calling CloudServices Support. My name is Jordan. How may I assist you with your account today?"
    },
    "context_data": {
      "portal_url": "https://portal.cloudservices.example.com"
    },
    "spoken_dialogue_rules": true
  },
  "voice": {
    "provider": "gemini",
    "voice_id": "Aoede",
    "language": "en-US",
    "speed": 1.0
  },
  "vad": {
    "provider": "native",
    "disabled": false,
    "silence_duration_ms": 1400,
    "prefix_padding_ms": 300,
    "threshold": 0.5
  },
  "turn_detection": {
    "mode": "server_vad",
    "allow_interruptions": true,
    "min_interruption_duration_ms": 400,
    "min_interruption_words": 2,
    "resume_false_interruption": true,
    "min_endpointing_delay_ms": 500,
    "user_away_timeout_seconds": 25.0
  },
  "session": {
    "max_duration_seconds": 1800,
    "inactivity_timeout_seconds": 180,
    "participant_join_timeout_seconds": 45.0,
    "delete_room_on_terminate": true,
    "recording": {
      "enabled": false,
      "audio": true,
      "traces": true
    }
  },
  "tools": [
    {
      "name": "lookup_account",
      "type": "webhook",
      "description": "Look up customer account subscription status, active services, and open support tickets using customer email or ID.",
      "parameters": {
        "type": "object",
        "properties": {
          "customer_id": {
            "type": "string",
            "description": "Customer email address or account ID"
          }
        },
        "required": ["customer_id"]
      },
      "config": {
        "url": "https://api.example.com/v1/support/accounts/lookup",
        "method": "POST",
        "headers": {
          "Authorization": "Bearer {{env:SUPPORT_API_KEY}}",
          "Content-Type": "application/json"
        },
        "timeout_seconds": 5.0
      }
    },
    {
      "name": "end_call",
      "type": "builtin",
      "description": "Gracefully terminates the phone call when support issue is resolved.",
      "config": {
        "delete_room": true,
        "drain_timeout_seconds": 3.0,
        "farewell_instruction": "Thank the caller for contacting CloudServices and wish them a wonderful day before disconnecting."
      }
    }
  ],
  "reporting": {
    "enabled": true,
    "endpoint": {
      "url": "https://api.example.com/v1/telemetry/calls",
      "method": "POST",
      "headers": {
        "Authorization": "Bearer {{env:TELEMETRY_KEY}}"
      },
      "timeout_seconds": 10.0
    },
    "retry_policy": {
      "max_retries": 3,
      "initial_delay_ms": 1000,
      "backoff_factor": 2.0
    },
    "security": {
      "validate_ssrf": true,
      "allow_local": false
    },
    "payload_config": {
      "include_chat_history": true,
      "include_model_usage": true,
      "include_events": true,
      "custom_fields": {
        "department": "tier1_support",
        "tenant_id": "tenant-corp-001"
      }
    }
  },
  "metadata": {
    "channel": "inbound_sip",
    "queue": "cloud_support_priority"
  }
}
```

---

### 7.4 Production Application Blueprint: Universal Appointment Scheduler

Available in repository at: `examples/appointment_booking/booking_metadata.json`

- **Scenario**: Outbound or inbound clinic, salon, or professional service appointment scheduling.
- **Key Features**: Deterministic sampling (`temperature: 0.5`), two sequential webhook tools (`check_availability` and `book_appointment`), dynamic business hours and timezone interpolation.

```json
{
  "version": "1.0",
  "agent": {
    "name": "universal-booking-agent",
    "role": "scheduler",
    "description": "Universal Appointment Booking Agent for Clinics, Salons, and Spas"
  },
  "models": {
    "provider": "gemini",
    "model": "gemini-3.8-live",
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    "modalities": ["audio"],
    "thinking": {
      "enabled": false,
      "reasoning_effort": "minimal",
      "include_thoughts": false
    },
    "custom_vocabulary": [
      "consultation",
      "reschedule",
      "booking",
      "practitioner",
      "confirmation code"
    ]
  },
  "instructions": {
    "system_prompt": "You are Taylor, an appointment scheduling coordinator for {business_name} in {timezone}. Your purpose is to assist callers in booking new appointments, checking open time slots, and rescheduling existing bookings. First ask what service they need and their preferred date or time window. Use the check_availability tool to verify openings. Once a slot is agreed upon, collect the caller's full name and phone number, then call the book_appointment tool. Confirm the details back to the caller before concluding.",
    "first_turn": {
      "speaker": "agent",
      "mode": "say",
      "text": "Hello! Thank you for calling scheduling services. My name is Taylor. What type of appointment can I help you book today?"
    },
    "context_data": {
      "business_name": "Metro Wellness Center",
      "timezone": "America/New_York"
    },
    "spoken_dialogue_rules": true
  },
  "voice": {
    "provider": "gemini",
    "voice_id": "Puck",
    "language": "en-US",
    "speed": 1.05
  },
  "vad": {
    "provider": "native",
    "disabled": false,
    "silence_duration_ms": 1500,
    "prefix_padding_ms": 300,
    "threshold": 0.5
  },
  "turn_detection": {
    "mode": "server_vad",
    "allow_interruptions": true,
    "min_interruption_duration_ms": 500,
    "min_interruption_words": 2,
    "resume_false_interruption": true,
    "min_endpointing_delay_ms": 500,
    "user_away_timeout_seconds": 20.0
  },
  "session": {
    "max_duration_seconds": 1200,
    "inactivity_timeout_seconds": 120,
    "participant_join_timeout_seconds": 30.0,
    "delete_room_on_terminate": true,
    "recording": {
      "enabled": false,
      "audio": true,
      "traces": true
    }
  },
  "tools": [
    {
      "name": "check_availability",
      "type": "webhook",
      "description": "Check available calendar appointment slots for a requested service type and date range.",
      "parameters": {
        "type": "object",
        "properties": {
          "service_type": {
            "type": "string",
            "description": "The category or type of service requested (e.g., consultation, therapy, styling)"
          },
          "date": {
            "type": "string",
            "description": "Preferred date in YYYY-MM-DD format"
          }
        },
        "required": ["service_type", "date"]
      },
      "config": {
        "url": "https://api.example.com/v1/appointments/availability",
        "method": "POST",
        "headers": {
          "Authorization": "Bearer {{env:BOOKING_API_KEY}}",
          "Content-Type": "application/json"
        },
        "timeout_seconds": 5.0
      }
    },
    {
      "name": "book_appointment",
      "type": "webhook",
      "description": "Finalize and book a confirmed appointment slot for the customer.",
      "parameters": {
        "type": "object",
        "properties": {
          "customer_name": {"type": "string", "description": "Full name of the customer"},
          "phone": {"type": "string", "description": "Contact phone number"},
          "service_type": {"type": "string", "description": "Service booked"},
          "slot_time": {"type": "string", "description": "ISO-8601 appointment start timestamp"}
        },
        "required": ["customer_name", "phone", "service_type", "slot_time"]
      },
      "config": {
        "url": "https://api.example.com/v1/appointments/book",
        "method": "POST",
        "headers": {
          "Authorization": "Bearer {{env:BOOKING_API_KEY}}",
          "Content-Type": "application/json"
        },
        "timeout_seconds": 5.0
      }
    },
    {
      "name": "end_call",
      "type": "builtin",
      "description": "Gracefully terminates the phone call once the appointment is booked or the caller wishes to hang up.",
      "config": {
        "delete_room": true,
        "drain_timeout_seconds": 3.0,
        "farewell_instruction": "Confirm the booked appointment date and time, thank the caller, and say goodbye."
      }
    }
  ],
  "reporting": {
    "enabled": true,
    "endpoint": {
      "url": "https://api.example.com/v1/telemetry/bookings",
      "method": "POST",
      "headers": {
        "Authorization": "Bearer {{env:TELEMETRY_KEY}}"
      },
      "timeout_seconds": 10.0
    },
    "retry_policy": {
      "max_retries": 3,
      "initial_delay_ms": 1000,
      "backoff_factor": 2.0
    },
    "security": {
      "validate_ssrf": true,
      "allow_local": false
    },
    "payload_config": {
      "include_chat_history": true,
      "include_model_usage": true,
      "include_events": true,
      "custom_fields": {
        "clinic_branch": "metro_downtown"
      }
    }
  },
  "metadata": {
    "booking_channel": "automated_phone"
  }
}
```

---

### 7.5 Production Application Blueprint: Virtual Receptionist & Inbound Concierge

Available in repository at: `examples/virtual_receptionist/receptionist_metadata.json`

- **Scenario**: Front-desk corporate building receptionist handling visitor questions and hours.
- **Key Features**: Lightweight configuration, immediate greeting utterance (`mode: "say"`), shorter max duration watchdog (15 minutes).

```json
{
  "version": "1.0",
  "agent": {
    "name": "inbound-receptionist-agent",
    "role": "receptionist",
    "description": "Inbound Concierge & Front-Desk Receptionist Voice Agent"
  },
  "models": {
    "provider": "gemini",
    "model": "gemini-3.8-live",
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    "modalities": ["audio"],
    "thinking": {
      "enabled": false,
      "reasoning_effort": "minimal",
      "include_thoughts": false
    },
    "custom_vocabulary": [
      "concierge",
      "headquarters",
      "extension",
      "directory",
      "department"
    ]
  },
  "instructions": {
    "system_prompt": "You are Morgan, the front-desk concierge and receptionist for Horizon Tower located at {address}. Greet all callers with professional warmth. Help callers by answering general inquiries regarding building hours ({building_hours}), office locations, and conference room reservations. If a caller asks to speak to an employee or department, take their message or provide the appropriate department contact information. Keep responses concise, welcoming, and polite.",
    "first_turn": {
      "speaker": "agent",
      "mode": "say",
      "text": "Good day! Thank you for calling Horizon Tower. My name is Morgan. How may I direct your call today?"
    },
    "context_data": {
      "building_hours": "Monday through Friday from 7 AM to 7 PM",
      "address": "100 Horizon Boulevard, Suite 500"
    },
    "spoken_dialogue_rules": true
  },
  "voice": {
    "provider": "gemini",
    "voice_id": "Aoede",
    "language": "en-US",
    "speed": 1.0
  },
  "vad": {
    "provider": "native",
    "disabled": false,
    "silence_duration_ms": 1500,
    "prefix_padding_ms": 300,
    "threshold": 0.5
  },
  "turn_detection": {
    "mode": "server_vad",
    "allow_interruptions": true,
    "min_interruption_duration_ms": 500,
    "min_interruption_words": 2,
    "resume_false_interruption": true,
    "min_endpointing_delay_ms": 500,
    "user_away_timeout_seconds": 20.0
  },
  "session": {
    "max_duration_seconds": 900,
    "inactivity_timeout_seconds": 90,
    "participant_join_timeout_seconds": 30.0,
    "delete_room_on_terminate": true,
    "recording": {
      "enabled": false,
      "audio": true,
      "traces": true
    }
  },
  "tools": [
    {
      "name": "end_call",
      "type": "builtin",
      "description": "Gracefully ends the phone call when caller inquiry has been answered or caller says goodbye.",
      "config": {
        "delete_room": true,
        "drain_timeout_seconds": 3.0,
        "farewell_instruction": "Thank the caller for reaching out to Horizon Tower and wish them a wonderful day."
      }
    }
  ],
  "reporting": {
    "enabled": true,
    "endpoint": {
      "url": "https://api.example.com/v1/telemetry/concierge",
      "method": "POST",
      "headers": {
        "Authorization": "Bearer {{env:TELEMETRY_KEY}}"
      },
      "timeout_seconds": 10.0
    },
    "retry_policy": {
      "max_retries": 3,
      "initial_delay_ms": 1000,
      "backoff_factor": 2.0
    },
    "security": {
      "validate_ssrf": true,
      "allow_local": false
    },
    "payload_config": {
      "include_chat_history": true,
      "include_model_usage": true,
      "include_events": true,
      "custom_fields": {
        "location": "Horizon Tower Lobby"
      }
    }
  },
  "metadata": {
    "facility": "horizon_hq"
  }
}
```

---

## 8. Error Handling, Safety Guards & Troubleshooting

### 8.1 Diagnosing `MetadataValidationError`

If a dispatched metadata payload violates the `SessionConfig` contract, the worker rejects the job before opening audio tracks and logs a structured diagnostic:

```text
agent_framework.config.validator.MetadataValidationError: Metadata validation failed with 2 error(s):
  - models.model: Field required
  - instructions.first_turn.text: first_turn.text must be provided and non-empty when mode is 'say'
```

#### Common Schema Validation Traps:
1. **Extra Unknown Fields**:
   - *Error*: `Extra inputs are not permitted [type=extra_forbidden]`
   - *Cause*: Placing unrecognized properties inside schema sections (e.g., `"model_name"` instead of `"model"`), or declaring nested properties at the root level (e.g., `"recording": {...}` or `"first_turn": {...}` placed at the root instead of inside `"session"` or `"instructions"`).
   - *Fix*: Check the field names against the architecture hierarchy in Section 3.2.
2. **Missing Mandatory Sections**:
   - *Error*: `Field required [type=missing, input_value=...]`
   - *Cause*: Omitting one of the 3 required sections: `models`, `instructions`, or `voice`.
3. **Invalid Turn Mode Pairing**:
   - *Error*: `Value error, first_turn.text must be provided and non-empty when mode is 'say'`
   - *Cause*: Specifying `"mode": "say"` without providing a valid string in `"text"`.

### 8.2 SSRF Guard Violations

If a webhook tool or reporting endpoint attempts to connect to a blocked IP or cloud metadata service:
```text
agent_framework.tools.webhook_tool.SSRFSecurityError: Webhook target URL 'http://169.254.169.254/latest/meta-data/' violates SSRF security policy: Cloud metadata IP address '169.254.169.254' is forbidden
```
- **Remedy**: Webhook URLs must resolve to publicly routable, valid domains or external IPs. If testing against a local mock server in development, set `"security": {"allow_local": true}`.

### 8.3 WebSocket Timeout, WS 1007 & 300-Second Graceful Call Draining

When managing worker containers in Kubernetes or Docker Compose during zero-downtime rolling updates:
1. Orchestrator sends `SIGTERM` to the worker process.
2. The worker immediately marks its health probe as draining and **refuses new room dispatches**.
3. Existing active calls continue uninterrupted up to `DRAIN_TIMEOUT_SECONDS` (default: 300s / 5 minutes).
4. As each room naturally concludes, its `end_call` audio playout drains, reporting webhooks fire, and the session releases cleanly.
5. Once all active sessions reach 0, the worker process exits with code 0.

### 8.4 Quick Diagnostic Checklist

| Symptom | Root Cause | Immediate Verification / Fix |
| :--- | :--- | :--- |
| **Worker fails to boot** | Missing required environment variables. | Run `uv run python deploy/verify_stack.py` to test SFU and Gemini API reachability. Ensure `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `GOOGLE_API_KEY` are populated. |
| **Agent joins room but exits immediately** | Invalid metadata JSON payload. | Inspect worker logs for `MetadataValidationError`. Check that all required fields (`models`, `instructions`, `voice`) are present and no extra fields exist at root. |
| **Agent speaks first word then cuts off** | `end_call` invoked without drain delay. | Ensure `end_call` config includes `"drain_timeout_seconds": 3.0` or higher. |
| **Long calls (>15m) disconnect with WS 1007** | Audio context explosion. | Verify context window compression is active (Gemini Live adapter automatically provisions a 50k sliding window). |
| **Webhook fails with SecretResolutionError** | Required environment variable not defined. | Ensure any variable referenced as `{{env:KEY}}` exists in the worker's `.env` or container environment. |
| **Model ignores specialized terms** | Speech recognition phonetic ambiguity. | Add domain-specific terms and acronyms to `models.custom_vocabulary`. |
