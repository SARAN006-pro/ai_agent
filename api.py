"""Minimal FastAPI layer for agent access."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import initialize_runtime, run_agent, get_last_metrics
from memory import save_memory
from logger import AgentLogger


app = FastAPI(title="Agent API", version="1.0.0")
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_READY = False
_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_API_LOG = AgentLogger("API")


def _cors_origins_from_env() -> tuple[list[str], bool]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    if not raw:
        return ["*"], False

    if raw == "*":
        # Browsers disallow allow_credentials=True with wildcard origin.
        return ["*"], False

    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if not origins:
        return ["*"], False
    return origins, True


_CORS_ORIGINS, _CORS_ALLOW_CREDENTIALS = _cors_origins_from_env()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=_CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class AnalyzeImageResponse(BaseModel):
    analysis: dict
    insights: str
    resources: list[dict]


def _is_incomplete_response(text: str) -> bool:
    t = (text or "").rstrip()
    if not t:
        return True
    if t.endswith(":") or t.endswith("-") or t.endswith("..."):
        return True
    if re.search(r"\n\s*(?:[-*]|\d+\.)\s*$", t):
        return True
    if len(t) > 80 and re.search(r"[A-Za-z0-9]$", t) and not re.search(r"[.!?\]\)\"]$", t):
        return True
    return False


def _extract_resources_from_search(search_output: dict, max_items: int = 4) -> list[dict]:
    items = search_output.get("results") if isinstance(search_output, dict) else None
    if not isinstance(items, list):
        return []

    resources: list[dict] = []
    seen_links: set[str] = set()
    for item in items[:max_items]:
        if not isinstance(item, dict):
            continue
        link = str(item.get("url", "")).strip()
        canonical = re.sub(r"^https?://", "", link, flags=re.IGNORECASE).rstrip("/").lower()
        if canonical and canonical in seen_links:
            continue
        if canonical:
            seen_links.add(canonical)
        resources.append(
            {
                "title": str(item.get("title", "Untitled")).strip(),
                "link": link,
                "summary": str(item.get("snippet", "")).strip(),
            }
        )
    return resources


async def _run_image_pipeline(file: UploadFile) -> AnalyzeImageResponse:
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    content_type = (file.content_type or "").lower()
    if content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid image type. Use jpg, jpeg, or png.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 8MB.")

    encoded = base64.b64encode(data).decode("ascii")
    image_data_url = f"data:{content_type};base64,{encoded}"

    agent = initialize_runtime()

    analysis = await asyncio.to_thread(agent.llm.vision_analyze, image_data_url)

    objects = analysis.get("objects", []) if isinstance(analysis, dict) else []
    text = str(analysis.get("text", "") if isinstance(analysis, dict) else "")
    description = str(analysis.get("description", "") if isinstance(analysis, dict) else "")

    resource_topic = " ".join([str(obj) for obj in objects[:3] if str(obj).strip()]).strip()
    if not resource_topic:
        resource_topic = text[:80].strip() or description[:80].strip()

    resources: list[dict] = []
    if resource_topic:
        search_tool = agent.registry.get("web_search")
        if search_tool is not None:
            query = f"{resource_topic} tutorial explanation product guide"
            search_result = await asyncio.to_thread(search_tool.run, query=query)
            if search_result.success:
                resources = _extract_resources_from_search(
                    search_result.output if isinstance(search_result.output, dict) else {},
                    max_items=4,
                )

    insight_prompt = (
        "You are helping with image understanding. "
        "Given this image-analysis JSON, provide concise actionable insights in 4-6 bullet points. "
        "Mention practical next steps and what the user can learn from the detected content.\n\n"
        f"Analysis JSON: {json.dumps(analysis)}"
    )
    insights = await asyncio.to_thread(run_agent, insight_prompt)

    return AnalyzeImageResponse(analysis=analysis, insights=insights, resources=resources)


def _ensure_runtime() -> None:
    """Lazy-safe runtime setup so API process can start even before first successful model init."""
    global _RUNTIME_READY
    if _RUNTIME_READY:
        return

    with _RUNTIME_LOCK:
        if _RUNTIME_READY:
            return
        initialize_runtime()
        _RUNTIME_READY = True


@app.get("/")
def serve_ui():
    index_file = _FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))

    return {
        "message": "Frontend is served by Next.js. Run it with 'npm run dev' inside the frontend directory.",
        "frontend_url": "http://127.0.0.1:3000",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    started_at = time.perf_counter()
    try:
        _ensure_runtime()
        response = await asyncio.to_thread(run_agent, message)
        save_memory(message, response)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        metrics = get_last_metrics()
        _API_LOG.info(
            f"/chat latency_ms={elapsed_ms:.2f} mode={metrics.get('mode')} "
            f"tool_calls={metrics.get('tool_calls', 0)} cache_hit={metrics.get('cache_hit', False)}"
        )
        return ChatResponse(response=response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"response": "Something went wrong"})


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest):
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    _ensure_runtime()
    started_at = time.perf_counter()
    response_text = await asyncio.to_thread(run_agent, message)
    save_memory(message, response_text)
    metrics = get_last_metrics()

    async def event_gen():
        chunk_size = 20
        for i in range(0, len(response_text), chunk_size):
            piece = response_text[i : i + chunk_size]
            payload_obj = {"type": "chunk", "content": piece}
            yield f"data: {json.dumps(payload_obj)}\n\n"
            await asyncio.sleep(0)

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        done_obj = {
            "type": "done",
            "latency_ms": round(elapsed_ms, 2),
            "mode": metrics.get("mode"),
            "tool_calls": metrics.get("tool_calls", 0),
            "cache_hit": metrics.get("cache_hit", False),
            "response_complete": not _is_incomplete_response(response_text),
        }
        _API_LOG.info(
            f"/chat/stream latency_ms={elapsed_ms:.2f} mode={metrics.get('mode')} "
            f"tool_calls={metrics.get('tool_calls', 0)} cache_hit={metrics.get('cache_hit', False)}"
        )
        yield f"data: {json.dumps(done_obj)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image(image: UploadFile = File(...)):
    started_at = time.perf_counter()
    try:
        _ensure_runtime()
        result = await _run_image_pipeline(image)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        _API_LOG.info(f"/analyze-image latency_ms={elapsed_ms:.2f}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        _API_LOG.error(f"/analyze-image failed: {e}")
        raise HTTPException(status_code=500, detail="Image analysis failed")
