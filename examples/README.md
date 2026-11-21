# LiveKit Agent Framework - Application Blueprints & Dispatch Guide

This directory contains production-ready, domain-neutral blueprints illustrating how to configure and dispatch voice agents using the LiveKit Agent Framework.

## 1. Blueprint Overview

| Blueprint | Directory | Description | Key Features |
| :--- | :--- | :--- | :--- |
| **Tier 1 Customer Support** | [`customer_support/`](./customer_support/support_metadata.json) | Technical & billing inquiries | Thinking model enabled, REST lookup webhook, custom vocabulary, `end_call` tool |
| **Appointment Booking** | [`appointment_booking/`](./appointment_booking/booking_metadata.json) | Clinic, salon & professional scheduling | Webhook availability checks, booking confirmation tool, context interpolation |
| **Virtual Receptionist** | [`virtual_receptionist/`](./virtual_receptionist/receptionist_metadata.json) | Inbound concierge & front-desk routing | Immediate greeting utterance, FAQ answering, call conclusion |

All blueprints strictly conform to the `SessionConfig` contract schema.

---

## 2. Running the Agent Framework Worker

Ensure environment variables are configured in `.env`:
```bash
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret
GOOGLE_API_KEY=your-gemini-api-key
```

Start the worker locally:
```bash
uv run agent-framework start
```
or in development mode with live reloading:
```bash
uv run agent-framework dev
```

---

## 3. Dispatching Sessions via LiveKit CLI

### Using LiveKit CLI (`lk`)
You can dispatch a voice agent with dynamic metadata using the LiveKit CLI tool:

#### Option A: Explicit Agent Dispatch (`lk dispatch create`)
Dispatches an agent worker directly into an active room with job metadata:

```bash
# Dispatch Tier 1 Support Agent
lk dispatch create \
  --room "support-call-101" \
  --agent-name "voice-agent" \
  --metadata "$(cat examples/customer_support/support_metadata.json)"

# Dispatch Appointment Booking Agent
lk dispatch create \
  --room "booking-call-202" \
  --agent-name "voice-agent" \
  --metadata "$(cat examples/appointment_booking/booking_metadata.json)"

# Dispatch Virtual Receptionist Agent
lk dispatch create \
  --room "receptionist-call-303" \
  --agent-name "voice-agent" \
  --metadata "$(cat examples/virtual_receptionist/receptionist_metadata.json)"
```

#### Option B: Room Creation with Metadata (`lk room create` & `lk room update`)
```bash
# 1. Create the room
lk room create support-call-101 --empty-timeout 300

# 2. Attach metadata to the room
lk room update support-call-101 --metadata "$(cat examples/customer_support/support_metadata.json)"
```

When a user joins `support-call-101`, the worker automatically parses the room metadata, validates the schema, initializes the Gemini Live model session with the specified prompt and tools, and speaks the initial greeting.

---

## 4. Dispatching Sessions via Backend API

### Python (`livekit-api`)
```python
import asyncio
import json
from pathlib import Path
from livekit import api

async def dispatch_voice_agent(room_name: str, blueprint_path: str):
    # Load blueprint
    with open(blueprint_path, "r", encoding="utf-8") as f:
        metadata_dict = json.load(f)

    # Optional: dynamically override fields at call time
    metadata_dict["metadata"]["caller_id"] = "+1-555-0199"
    metadata_dict["instructions"]["context_data"]["customer_name"] = "Alice Smith"

    lk_api = api.LiveKitAPI()
    room = await lk_api.room.create_room(
        api.CreateRoomRequest(
            name=room_name,
            empty_timeout=300,
            metadata=json.dumps(metadata_dict),
        )
    )
    print(f"Dispatched agent to room: {room.name}")
    await lk_api.aclose()

if __name__ == "__main__":
    asyncio.run(dispatch_voice_agent("call-room-alice", "examples/customer_support/support_metadata.json"))
```

### Node / TypeScript (`livekit-server-sdk`)
```typescript
import { RoomServiceClient } from 'livekit-server-sdk';
import * as fs from 'fs';

const livekitHost = process.env.LIVEKIT_URL || 'http://127.0.0.1:7880';
const svc = new RoomServiceClient(livekitHost, process.env.LIVEKIT_API_KEY, process.env.LIVEKIT_API_SECRET);

async function createAgentRoom(roomName: string, blueprintPath: string) {
  const metadata = JSON.parse(fs.readFileSync(blueprintPath, 'utf-8'));

  // Customize dynamic caller context
  metadata.metadata = { caller_id: '+1-555-0199' };

  const room = await svc.createRoom({
    name: roomName,
    emptyTimeout: 300,
    metadata: JSON.stringify(metadata),
  });

  console.log(`Created room ${room.name} with agent metadata`);
}

createAgentRoom('call-room-alice', 'examples/appointment_booking/booking_metadata.json');
```

---

## 5. Estimating Costs for Blueprints

You can use `scripts/estimate_sts_cost.py` to calculate exact token consumption, LiveKit participant minutes, and blended cost per call for any blueprint:

```bash
# Calculate costs for Customer Support blueprint (5-minute call)
uv run python scripts/estimate_sts_cost.py --blueprint customer_support --duration 5

# Calculate costs for Appointment Booking blueprint (3-minute call)
uv run python scripts/estimate_sts_cost.py --blueprint appointment_booking --duration 3

# Calculate costs for Virtual Receptionist blueprint (2-minute call)
uv run python scripts/estimate_sts_cost.py --blueprint virtual_receptionist --duration 2
```

---

## 6. Secret References (`{{env:KEY}}`)

Notice that webhook headers and reporting secrets use `{{env:KEY}}` notation:
```json
"headers": {
  "Authorization": "Bearer {{env:SUPPORT_API_KEY}}"
}
```
At runtime, the worker securely resolves `{{env:SUPPORT_API_KEY}}` from the host environment variables. This prevents hardcoding sensitive API tokens in metadata payloads passed across the network.
