# Autonomous ReAct Agent System — OpenRouter Edition

Same production-quality agent as before, now powered by **OpenRouter**
instead of the Anthropic SDK directly. One API key, access to 100+
models including Claude, GPT-4o, Llama, Mistral, and more.

---

## Setup

```bash
pip install openai
export OPENROUTER_API_KEY=sk-or-...   # from https://openrouter.ai/keys
cd agent_system
python main.py
```

Get your free OpenRouter key at **https://openrouter.ai/keys**
(Free credits on signup. Claude models cost credits.)

---

## Windows Run Guide (Backend + Frontend)

Run these commands from the project root.

### 1. One-time setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 2. Configure environment

Create a `.env` file in the project root and add:

```env
OPENROUTER_API_KEY=your_key_here
```

### 3. Start both services with one command

```powershell
.\start-dev.ps1
```

Defaults:
- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:3001`

Use custom ports if needed:

```powershell
.\start-dev.ps1 -BackendPort 8001 -FrontendPort 3002
```

### 4. Manual start (optional)

Backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Frontend (new terminal):

```powershell
cd frontend
npm run dev -- --port 3001
```

### 5. Verify services

- Backend health: `http://127.0.0.1:8000/health` (expected `{"status":"ok"}`)
- Frontend: `http://127.0.0.1:3001`

### 6. Troubleshooting

- Port in use: change ports (frontend `3002`, backend `8001`) or stop the process using the port.
- Backend startup issue: reactivate `.venv` and reinstall deps with `pip install -r requirements.txt`.
- Image upload endpoint issue: ensure `python-multipart` is installed:

```powershell
pip install python-multipart
```

---

## Deploy (Render Backend + Vercel Frontend)

This repo is best deployed as:
- Backend: Render (Python web service)
- Frontend: Vercel (Next.js)

### 1. Backend on Render

The backend entrypoint is already configured in `render.yaml`:

```yaml
startCommand: "uvicorn api:app --host 0.0.0.0 --port $PORT"
```

On Render dashboard:
- Runtime: Python
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

Set environment variables in Render:
- `OPENROUTER_API_KEY=your_key_here`
- `CORS_ALLOW_ORIGINS=https://your-frontend.vercel.app`

After deploy, verify:
- `https://your-backend.onrender.com/health`

### 2. Frontend on Vercel (Next.js)

Import the repo in Vercel and select the `frontend` directory as the project root.

Recommended settings:
- Framework preset: Next.js
- Build command: `npm run build`
- Output: default Next.js output

Set frontend env var in Vercel:
- `NEXT_PUBLIC_API_BASE_URL=https://your-backend.onrender.com`

The frontend now reads this env var for:
- `/chat`
- `/chat/stream`
- `/analyze-image`

### 3. Local frontend env setup

Copy `frontend/.env.example` to `frontend/.env.local` and set:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

### 4. Push and redeploy

```bash
git add .
git commit -m "deploy: configure render + vercel integration"
git push
```

### 5. Final verification

- Open your Vercel frontend URL
- Send a chat message
- Confirm network calls reach `https://your-backend.onrender.com`
- Confirm backend responses return successfully

---

## Switch models instantly

In `main.py`, change one line:

```python
llm = LLMClient(model="anthropic/claude-opus-4-5")   # default
llm = LLMClient(model="anthropic/claude-sonnet-4-5") # faster, cheaper
llm = LLMClient(model="openai/gpt-4o")               # OpenAI
llm = LLMClient(model="meta-llama/llama-3.1-70b-instruct")  # free tier
```

Zero other code changes needed.

---

## Architecture

```
main.py                     ← wires components, runs demos
│
├── core/
│   ├── agent.py            ← ReAct reasoning loop
│   ├── llm.py              ← OpenRouter API client (openai SDK)
│   └── tools.py            ← BaseTool contract + ToolRegistry
│
├── tools/
│   └── implementations.py  ← 5 tools: search, calculator, explainer,
│                              text analyzer, study plan generator
│
└── utils/
    └── logger.py           ← Structured observability logger
```

---

## Why OpenRouter instead of Anthropic SDK directly?

| | Anthropic SDK | OpenRouter |
|---|---|---|
| Models | Claude only | 100+ models |
| API format | Anthropic-specific | OpenAI-compatible |
| Key management | Per-provider keys | Single key |
| Cost comparison | Direct pricing | Same + small markup |
| Free tier | No | Yes (some models) |

For an interview, OpenRouter is a strong demonstration: you understand
that the underlying model is an implementation detail, not the system.

---

## How the ReAct loop works

```
User task
    │
    ▼
[Iteration N]
  Model sees: system prompt + tool descriptions + full conversation history
  Returns JSON: { thought, action, tool_name, tool_input }
    │
    ▼
  Agent executes tool → ToolResult(success, output, error)
  Result appended to conversation history
    │
    ▼
  Loop continues until action = "final_answer" or MAX_ITERATIONS hit
```

---

## Robustness mechanisms

| Risk | Protection |
|------|-----------|
| Model returns invalid JSON | Retry with correction nudge (3x) |
| Model calls non-existent tool | Error injected, model recovers |
| Tool execution fails | ToolResult(success=False), model reasons about it |
| Same tool called in a loop | Detected after 3 identical calls |
| Task never completes | MAX_ITERATIONS = 12 hard cap |
| Rate limit | Exponential backoff retry |

---

## Interview talking points

**"Walk me through your architecture."**
> "I separated concerns into three layers: the LLM client handles all
> API calls and retries behind a clean interface; the tool registry is
> the single source of truth for available tools; the agent loop is
> pure decision logic. I switched from the Anthropic SDK to OpenRouter
> so the same code can run any model."

**"Why OpenRouter?"**
> "OpenRouter gives me a single OpenAI-compatible API for 100+ models.
> I can benchmark Claude vs GPT-4o vs Llama on the same tasks by
> changing one string. In production that's valuable for cost and
> performance optimization."

**"How do you handle tool failures?"**
> "Tools never raise exceptions — they return ToolResult(success=False)
> with an error description. That goes back into the conversation, so
> the model can reason about what went wrong and try something different."
