# LiveKit Voice Agent Framework - Production Operations Guide

This guide details the deployment, scaling, security hardening, and operational runbooks for running the **LiveKit Voice Agent Framework** in enterprise production environments.

---

## 1. Architecture & Networking Overview

LiveKit agent workers operate on a **job-dispatch worker model**:

```
                       +---------------------------------------+
                       |          Caller / Human User          |
                       |   (Browser / SIP Telephony / WebRTC)  |
                       +-------------------+-------------------+
                                           | WebRTC Audio (Opus / PCM)
                                           v
                       +---------------------------------------+
                       |         LiveKit SFU / Cloud           |
                       |    (Room State & Job Dispatching)     |
                       +-------------------+-------------------+
                                           |
                                           | Outbound WebSocket
                                           v
                       +---------------------------------------+
                       |     LiveKit Agent Framework Worker    |
                       |    (Container / VM on Port 8081)      |
                       +---------+-------------------+---------+
                                 |                   |
               Bidirectional STS |                   | Telemetry
            (16kHz in / 24kHz out)                   | (SSRF Guarded Webhook)
                                 v                   v
              +-----------------------+   +-----------------------+
              | Google Gemini 3.8 Live|   |   Your Backend API /  |
              |  (or OpenAI Realtime) |   |   Analytics Webhook   |
              +-----------------------+   +-----------------------+
```

### Key Security & Network Characteristics
- **Outbound-Only SFU Connectivity**: Agent workers connect **outbound** to LiveKit SFU via WebSocket (`LIVEKIT_URL`). Workers require **zero public inbound ports** or incoming NAT configuration.
- **Internal-Only Health Endpoint**: The worker exposes an internal HTTP server on port 8081 (`http://127.0.0.1:8081/`) responding with HTTP 200 OK for Docker / Kubernetes health probes.
- **Graceful Call Draining**: When receiving a `SIGTERM` signal, the worker immediately rejects new incoming room dispatches while preserving all active calls up to `DRAIN_TIMEOUT_SECONDS` (default: 300 seconds / 5 minutes).

---

## 2. Deployment Topology Options

### Option A: Worker-Only Deployment (LiveKit Cloud or Remote SFU)

Use `docker-compose.yml` to run the agent worker against LiveKit Cloud or an existing self-hosted cluster.

1. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```
   Populate credentials in `.env`:
   ```bash
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=your-api-key
   LIVEKIT_API_SECRET=your-api-secret
   GOOGLE_API_KEY=your-gemini-api-key
   ```

2. **Deploy Stack**:
   ```bash
   docker compose up -d --build
   ```

3. **Verify Health**:
   ```bash
   curl -f http://127.0.0.1:8081/
   docker compose ps
   ```

4. **View Logs**:
   ```bash
   docker compose logs -f agent-framework
   ```

---

### Option B: Full-Stack Self-Hosted Deployment (SFU + Redis + Worker)

Use `docker-compose.self-hosted.yml` to spin up a complete local or private cloud environment including:
- **Redis 7**: Distributed room state and agent dispatch queue.
- **LiveKit Server (SFU)**: Media routing, WebRTC TURN/ICE, and room orchestration.
- **Agent Framework Worker**: Voice AI processing container.

1. **Deploy Stack**:
   ```bash
   docker compose -f docker-compose.self-hosted.yml up -d --build
   ```

2. **Required Firewall Ports (Host Network)**:
   - `7880/tcp`: LiveKit HTTP / WebSocket API & room signaling.
   - `7881/tcp`: LiveKit TCP fallback media port.
   - `7882/udp`: LiveKit TURN UDP port.
   - `50000-50020/udp`: LiveKit WebRTC ICE UDP media port range.

3. **LiveKit Configuration**:
   The SFU configuration is located in `deploy/livekit.yaml`. For external production access, set `use_external_ip: true` and configure your public domain.

4. **Verify All Services**:
   ```bash
   docker compose -f docker-compose.self-hosted.yml ps
   ```

---

### Option C: Enterprise Kubernetes Deployment (`deploy/k8s/`)

Deploy to a Kubernetes cluster with zero-downtime rolling updates, Horizontal Pod Autoscaling (HPA), and Pod Disruption Budgets (PDB).

#### Manifest Structure
```
deploy/k8s/
├── namespace.yaml      # Dedicated 'livekit-agent-framework' namespace
├── configmap.yaml      # Non-sensitive configuration (LIVEKIT_URL, log levels, thresholds)
├── secret.yaml         # Sensitive API credentials (LIVEKIT_*, GOOGLE_API_KEY)
├── service.yaml        # Internal ClusterIP service exposing port 8081
├── deployment.yaml     # RollingUpdate worker deployment with 300s termination grace
├── pdb.yaml            # PodDisruptionBudget ensuring minAvailable: 1
└── hpa.yaml            # HPA scaling 2-10 replicas based on 50% CPU
```

#### Step-by-Step Installation
1. **Create Namespace**:
   ```bash
   kubectl apply -f deploy/k8s/namespace.yaml
   ```

2. **Configure Secrets & ConfigMap**:
   Edit `deploy/k8s/secret.yaml` with your actual production keys, then apply:
   ```bash
   kubectl apply -f deploy/k8s/secret.yaml
   kubectl apply -f deploy/k8s/configmap.yaml
   ```

3. **Deploy Worker Service & Pods**:
   ```bash
   kubectl apply -f deploy/k8s/service.yaml
   kubectl apply -f deploy/k8s/deployment.yaml
   ```

4. **Apply Resilience Policies (PDB & HPA)**:
   ```bash
   kubectl apply -f deploy/k8s/pdb.yaml
   kubectl apply -f deploy/k8s/hpa.yaml
   ```

5. **Verify Rollout Status**:
   ```bash
   kubectl rollout status deployment/agent-framework-worker -n livekit-agent-framework
   kubectl get pods,hpa,pdb -n livekit-agent-framework
   ```

#### Rolling Update & Draining Behavior
The deployment configures:
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 0
terminationGracePeriodSeconds: 300
```
When a new container image is rolled out, Kubernetes launches the new replica first (`maxSurge: 1`), waits for readiness on `:8081`, and issues `SIGTERM` to the old replica. The worker immediately marks itself as draining, completes existing calls up to 300 seconds, and terminates without abrupt user disconnection.

---

### Option D: Host Systemd Service (Linux VM)

For production deployments on dedicated Ubuntu, Debian, or RHEL instances.

1. **Copy Service Unit**:
   ```bash
   sudo cp deploy/systemd/agent-framework.service /etc/systemd/system/
   ```

2. **Set Up Project Directory**:
   Ensure `/opt/agent-framework` contains the repository and a configured `.env` file:
   ```bash
   sudo mkdir -p /opt/agent-framework
   # Deploy code and .env to /opt/agent-framework
   ```

3. **Enable and Start Service**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now agent-framework.service
   ```

4. **Service Management**:
   ```bash
   # Check service status
   sudo systemctl status agent-framework.service

   # View journal logs
   sudo journalctl -u agent-framework.service -f

   # Graceful restart (initiates 300s call drain)
   sudo systemctl restart agent-framework.service
   ```

---

## 3. Deployment Automation & Verification Tools

### Automated Rollout Script (`deploy/deploy.sh`)

Performs pre-flight environment checks, image builds, zero-downtime container rollout, and health probe verification:

```bash
chmod +x deploy/deploy.sh
./deploy/deploy.sh
```

**Workflow:**
1. Checks Docker daemon availability.
2. Validates `.env` file and required credentials (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `GOOGLE_API_KEY`).
3. Executes multi-stage image build.
4. Performs container recreation with `--remove-orphans`.
5. Polls local health check (`http://127.0.0.1:8081/`) up to 45 seconds.

### Stack Verification Tool (`deploy/verify_stack.py`)

Run diagnostics before deployment or in CI/CD pipelines:

```bash
# Full verification against active environment
uv run python deploy/verify_stack.py

# Offline pre-flight verification (skip live SFU / Gemini network calls)
uv run python deploy/verify_stack.py --skip-sfu --skip-gemini

# Validate a specific environment file
uv run python deploy/verify_stack.py --env-file /path/to/.env
```

---

## 4. Production Configuration Reference

All settings can be specified via `.env` or system environment variables:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `LIVEKIT_URL` | String | `ws://127.0.0.1:7880` | LiveKit SFU WebSocket endpoint (`wss://...`) |
| `LIVEKIT_API_KEY` | String | `""` | LiveKit Project API Key |
| `LIVEKIT_API_SECRET` | String | `""` | LiveKit Project API Secret |
| `GOOGLE_API_KEY` | String | `""` | Google Gemini API Key |
| `OPENAI_API_KEY` | String | `""` | Optional OpenAI API Key |
| `AGENT_NAME` | String | `livekit-voice-agent` | Agent identifier registered with LiveKit SFU |
| `HEALTH_CHECK_HOST` | String | `0.0.0.0` | Health check server bind address |
| `HEALTH_CHECK_PORT` | Integer | `8081` | Health check server port |
| `DRAIN_TIMEOUT_SECONDS` | Float | `300.0` | Grace period (in seconds) for active calls during shutdown |
| `WORKER_LOAD_THRESHOLD` | Float | `0.7` | CPU/load ratio (0.0–1.0) above which worker pauses new job dispatches |
| `PROMETHEUS_PORT` | Integer | `None` | Optional port for Prometheus metrics exporter |
| `LIVEKIT_LOG_LEVEL` | String | `INFO` | Worker logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `ALLOW_LOCAL_WEBHOOKS` | Boolean | `false` | SSRF security override allowing private IP webhook callbacks |

---

## 5. Troubleshooting & Operational Runbook

### Symptom: WebRTC Audio Failing to Connect (Self-Hosted)
- **Cause**: ICE UDP media ports or TURN UDP port blocked by host firewall.
- **Resolution**: Verify UDP ports `7882` and `50000-50020` are open on the host firewall / security group. Ensure `deploy/livekit.yaml` specifies the reachable external IP if running on a public cloud instance.

### Symptom: Deployment Timeout on Restart
- **Cause**: Active calls are still connected and draining under `DRAIN_TIMEOUT_SECONDS: 300`.
- **Resolution**: This is intended behavior. The worker waits up to 5 minutes for callers to finish before killing the container. Verify `stop_grace_period` in Compose and `terminationGracePeriodSeconds` in Kubernetes are set to at least 300s.

### Symptom: Worker Not Accepting New Calls
- **Cause**: CPU or memory utilization exceeded `WORKER_LOAD_THRESHOLD` (default: 0.70 / 70%).
- **Resolution**: Check CPU and memory usage using `docker stats` or `kubectl top pods`. Scale up worker replicas using `docker compose up -d --scale agent-framework=3` or inspect HPA metrics.

### Symptom: SSRF Block on Webhook Callback
- **Cause**: Dynamic tool webhook or post-call reporting URL targeted a loopback, private RFC-1918, or cloud metadata IP (`169.254.169.254`).
- **Resolution**: Configure target reporting URL to a valid public hostname or set `ALLOW_LOCAL_WEBHOOKS=true` only in trusted local test environments.
