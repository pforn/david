# Project David 🤖

**Project David** is a distributed, 3-node AI assistant designed for **high-speed local inference** and **agentic web orchestration**, operating over a Tailscale mesh network. 

## System Architecture

The architecture relies on specialized hardware nodes to separate computation from orchestration and sensory input.

| Node | Hardware | Role | Stack |
|------|----------|------|-------|
| **Node 1** | Xbox Series X (Dev Mode) | High-speed offline LLM inference (Phi-3 ONNX via DirectML). Unlocked memory sandbox (10GB VRAM) running as a Game title. | C# / UWP |
| **Node 2** | Lenovo Laptop (Ubuntu) | Central Orchestrator & router. Employs `browser-use` for web interactions and routes prompts between the local Xbox or the cloud (Claude 3.5 Sonnet) based on complexity. | Python / FastAPI / Docker |
| **Node 3** | M4 Mac & Pixel 7 Pro | Client endpoints providing sensory input via voice and text interfaces. | TBD |

For full details on the planned implementation, refer to the [Project Plan](Project_Plan.md).

---

## Node 1: Xbox AI Server 🎮

The Xbox Serve is a UWP application hosting a local `HttpListener` on port `8080`. 

### Key Features
* **Bypassed Sandbox**: Runs in Game Mode (`AppListEntry="Game"`) and has Full Trust Execution, unlocking the full power of the internal AMD GPU and 10GB of VRAM.
* **Anti-Suspension**: Extended execution lifecycles prevent the background server process from suspending.
* **On-Device Generative AI**: Uses `Microsoft.ML.OnnxRuntimeGenAI.DirectML` to perform inference entirely locally via an INT4-AWQ quantized Phi-3 Mini model.

### Getting Started with Node 1

**1. Download the Phi-3 Model**
Node 1 relies on the `microsoft/Phi-3-mini-4k-instruct-onnx` model (specifically the DirectML INT4-AWQ variant). We have provided a fast download script:

```bash
cd Xbox_AI_Server
uv run download_phi3.py
```
*(This pulls down ~2.4GB of weights into `Assets/Model/directml/directml-int4-awq-block-128/`)*

**2. Deploy to Xbox**
* Switch your Xbox Series X to **Dev Mode** and access the Xbox Device Portal from your PC.
* Compile the `DavidInference` project from Visual Studio 2022 (Deploy to Remote Machine).

**3. Test the API Endpoints**
Once running, ping the server using curl or Postman over the Tailscale network:

**Health Check:**
```bash
curl -X GET http://<xbox-tailscale-ip>:8080/health
```

**Run Inference:**
```bash
curl -X POST http://<xbox-tailscale-ip>:8080/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a joke about distributed systems.", "maxTokens": 256}'
```