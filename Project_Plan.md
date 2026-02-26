# Project Jarvis — System Architecture & Project Plan

## Overview

Jarvis is a distributed, 3-node AI assistant designed for **high-speed local inference** and **agentic web orchestration**. The system spans heterogeneous hardware connected via a Tailscale mesh network.

| Node | Hardware | Role | Stack |
|------|----------|------|-------|
| **Node 1** | Xbox Series X (Dev Mode) | Offline LLM inference (Phi-3 ONNX / DirectML) | C# / UWP |
| **Node 2** | Lenovo Laptop (Ubuntu Server) | Central router & agentic executor | Python / FastAPI / Docker |
| **Node 3** | M4 Mac & Pixel 7 Pro | Sensory input (Voice / Text) | TBD |

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    TAILSCALE MESH NETWORK                     │
│                                                              │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │   NODE 3    │    │     NODE 2       │    │   NODE 1   │  │
│  │  Clients    │───▶│   Orchestrator   │───▶│  Compute   │  │
│  │             │    │                  │    │  Engine    │  │
│  │ • M4 Mac    │    │ • FastAPI Router │    │            │  │
│  │ • Pixel 7   │    │ • Claude 3.5 API │    │ • Xbox     │  │
│  │             │    │ • browser-use    │    │ • Phi-3    │  │
│  │ Voice/Text  │    │ • Docker         │    │ • DirectML │  │
│  └─────────────┘    └──────────────────┘    └────────────┘  │
│                              │                               │
│                              ▼                               │
│                     ┌────────────────┐                       │
│                     │  Claude 3.5    │                       │
│                     │  Sonnet API    │                       │
│                     └────────────────┘                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Build Order

> **Phase 1 (Current Session): Node 1 — The Xbox Compute Engine**
>
> Phase 2: Node 2 — The Orchestrator
>
> Phase 3: Node 3 — The Clients
>
> Phase 4: Integration & end-to-end testing

---

## Directory Structure

```
jarvis/
│
├── README.md                          # Top-level project overview
├── Project_Plan.md                    # ← This file
│
├── node1-xbox/                        # ── NODE 1: COMPUTE ENGINE ──
│   │
│   ├── JarvisInference/               # UWP C# project root
│   │   ├── JarvisInference.sln        # Visual Studio solution
│   │   │
│   │   ├── JarvisInference/           # Main UWP app project
│   │   │   ├── App.xaml               # Application entry point
│   │   │   ├── App.xaml.cs
│   │   │   ├── MainPage.xaml          # Minimal status UI
│   │   │   ├── MainPage.xaml.cs
│   │   │   │
│   │   │   ├── Server/
│   │   │   │   └── HttpPromptServer.cs    # HttpListener on :8080
│   │   │   │
│   │   │   ├── Inference/
│   │   │   │   ├── OnnxModelLoader.cs     # Load quantized Phi-3 via ONNX Runtime
│   │   │   │   └── InferenceEngine.cs     # Tokenize → Run → Decode pipeline
│   │   │   │
│   │   │   ├── Models/
│   │   │   │   ├── PromptRequest.cs       # Incoming JSON schema
│   │   │   │   └── PromptResponse.cs      # Outgoing JSON schema
│   │   │   │
│   │   │   ├── Properties/
│   │   │   │   └── AssemblyInfo.cs
│   │   │   │
│   │   │   ├── Assets/                    # UWP required assets (icons, etc.)
│   │   │   │
│   │   │   └── Package.appxmanifest       # ⚠ MUST include AppListEntry="Game"
│   │   │                                  #   to unlock 10 GB VRAM sandbox
│   │   │
│   │   └── JarvisInference.Tests/         # Unit tests (optional xUnit/MSTest)
│   │       └── InferenceEngineTests.cs
│   │
│   ├── models/                            # Git-ignored; large model binaries
│   │   └── phi-3-mini-4bit/
│   │       ├── model.onnx
│   │       └── tokenizer.json
│   │
│   └── docs/
│       └── xbox-dev-mode-setup.md         # Step-by-step Xbox Dev Mode guide
│
├── node2-orchestrator/                # ── NODE 2: ORCHESTRATOR ──
│   │
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   │
│   ├── app/
│   │   ├── main.py                    # FastAPI entry point
│   │   ├── config.py                  # Env vars, API keys, node addresses
│   │   │
│   │   ├── router/
│   │   │   ├── prompt_router.py       # Route logic: Xbox vs Claude
│   │   │   └── health.py             # Health-check endpoints
│   │   │
│   │   ├── clients/
│   │   │   ├── xbox_client.py         # HTTP client → Node 1 :8080
│   │   │   └── claude_client.py       # Anthropic SDK wrapper
│   │   │
│   │   ├── agents/
│   │   │   ├── browser_agent.py       # browser-use autonomous browsing
│   │   │   └── job_scraper.py         # Job board scraper (Sales Eng / London)
│   │   │
│   │   └── pipelines/
│   │       └── macro_insights.py      # Financial data pipeline
│   │
│   └── tests/
│       └── test_router.py
│
├── node3-clients/                     # ── NODE 3: CLIENTS ──
│   │
│   ├── mac-client/                    # M4 Mac client (TBD)
│   │   └── README.md
│   │
│   └── pixel-client/                  # Pixel 7 Pro client (TBD)
│       └── README.md
│
├── shared/                            # Cross-node contracts
│   ├── schemas/
│   │   └── prompt_schema.json         # Canonical JSON schema for prompt payloads
│   └── tailscale/
│       └── network_setup.md           # Tailscale mesh configuration guide
│
└── .gitignore                         # Ignore models/, secrets, build artifacts
```

---

## Node 1 — Key Constraints & Notes

| Concern | Detail |
|---------|--------|
| **Memory Sandbox Bypass** | `Package.appxmanifest` must declare `AppListEntry="Game"` to access the full 10 GB VRAM allocation instead of the default UWP sandbox limit. |
| **Model Format** | Phi-3 Mini, 4-bit quantized, ONNX format. Inference via `Microsoft.ML.OnnxRuntime.DirectML` NuGet package. |
| **Network** | `HttpListener` bound to `http://*:8080/`. Xbox Dev Mode firewall must allow inbound TCP 8080. |
| **Deployment** | Sideloaded via Xbox Device Portal or Visual Studio remote deployment over the local network. |
| **Input Contract** | `POST /api/prompt` with JSON body `{ "prompt": "...", "max_tokens": 256 }` |
| **Output Contract** | JSON response `{ "response": "...", "tokens_used": 42, "inference_ms": 310 }` |

---

## Next Steps

- [ ] Scaffold the `node1-xbox/JarvisInference` UWP project
- [ ] Configure `Package.appxmanifest` with `AppListEntry="Game"`
- [ ] Implement `HttpPromptServer.cs` (receive prompts on :8080)
- [ ] Implement `OnnxModelLoader.cs` + `InferenceEngine.cs`
- [ ] Define `PromptRequest` / `PromptResponse` models
- [ ] Test locally before sideloading to Xbox
