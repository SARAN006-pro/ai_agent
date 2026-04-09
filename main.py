"""
Main entrypoint — OpenRouter version.

Run with:
    pip install openai
    export OPENROUTER_API_KEY=your_key_here
    cd agent_system
    python main.py

Get your OpenRouter API key at: https://openrouter.ai/keys
Free credits available on signup. Claude models require credits.
"""

import os
import sys
import re
import time
import json
import threading
from difflib import SequenceMatcher
from collections import OrderedDict

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

from llm import LLMClient
from tools import ToolRegistry
from agent import Agent
from tools import ToolResult
from logger import AgentLogger
from implementations import (
    WebSearchTool,
    QueryRefinerTool,
    CalculatorTool,
    ConceptExplainerTool,
    TextAnalyzerTool,
    StudyPlanTool,
)
from memory import init_db, save_memory, get_recent_memories


_AGENT: Agent | None = None
_LAST_PLAN: str = ""
_LAST_METRICS: dict = {}
_RUNTIME_LOG = AgentLogger("Runtime")

_CACHE_MAX_ITEMS = int(os.getenv("RESPONSE_CACHE_MAX_ITEMS", "200"))
_CACHE_TTL_SECONDS = int(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "600"))
_RESPONSE_CACHE: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
_SIMILARITY_THRESHOLD = float(os.getenv("REPEAT_QUERY_SIMILARITY_THRESHOLD", "0.8"))
_LAST_QUERY_NORMALIZED: str | None = None
_QUERY_STATE_LOCK = threading.Lock()

_FOLLOW_UP_SIMILARITY_THRESHOLD = float(os.getenv("FOLLOW_UP_QUERY_SIMILARITY_THRESHOLD", "0.45"))
_FOLLOW_UP_HINTS = (
    "again",
    "more",
    "another",
    "next",
    "also",
    "else",
    "different",
    "new",
    "further",
    "continue",
)


def _is_news_like_query(text: str) -> bool:
    normalized = _normalize_query(text)
    return any(token in normalized for token in ("news", "latest", "headlines", "updates", "today"))


def _classify_intent_state(task: str, last_query: str | None, similarity: float) -> str:
    if not last_query:
        return "new_topic"

    normalized = _normalize_query(task)
    if normalized == last_query or similarity >= _SIMILARITY_THRESHOLD:
        return "repeat_request"

    if similarity >= _FOLLOW_UP_SIMILARITY_THRESHOLD:
        return "follow_up"

    if any(hint in normalized for hint in _FOLLOW_UP_HINTS):
        return "follow_up"

    return "new_topic"


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _cache_get(query: str) -> str | None:
    key = _normalize_query(query)
    if not key:
        return None
    now = time.time()
    item = _RESPONSE_CACHE.get(key)
    if not item:
        return None

    created_at, value = item
    if now - created_at > _CACHE_TTL_SECONDS:
        _RESPONSE_CACHE.pop(key, None)
        return None

    _RESPONSE_CACHE.move_to_end(key)
    return value


def _cache_set(query: str, response: str) -> None:
    key = _normalize_query(query)
    if not key:
        return
    _RESPONSE_CACHE[key] = (time.time(), response)
    _RESPONSE_CACHE.move_to_end(key)
    while len(_RESPONSE_CACHE) > _CACHE_MAX_ITEMS:
        _RESPONSE_CACHE.popitem(last=False)


def _query_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _build_repeat_response(task: str, cached_response: str) -> str:
    """Refine a cached answer for repeat prompts without re-running tools."""
    # If cached response is structured search data, keep message separate from articles.
    try:
        parsed = json.loads(cached_response)
        if isinstance(parsed, dict) and isinstance(parsed.get("articles"), list):
            structured = {
                "message": "You've already asked this. Here's a refined summary.",
                "insights": parsed.get("insights", []),
                "articles": parsed.get("articles", []),
            }
            return json.dumps(structured, ensure_ascii=False)
    except Exception:
        pass

    if _AGENT is None:
        return (
            "You've already asked this. Here's a refined summary.\n\n"
            f"{cached_response}"
        )

    prompt = (
        "The user repeated or closely rephrased a previous query. "
        "Do not perform web search or call tools. "
        "Return only a refined answer with:\n"
        "1) concise summary of the previous answer\n"
        "2) one new angle or deeper insight\n"
        "3) one practical next step\n\n"
        f"Current query: {task}\n\n"
        f"Previous answer:\n{cached_response}"
    )

    try:
        refined = _AGENT.quick_answer(prompt)
        refined_text = (refined or "").strip()
        if refined_text:
            return "You've already asked this. Here's a refined summary.\n\n" + refined_text
    except Exception:
        pass

    return "You've already asked this. Here's a refined summary.\n\n" + cached_response


def _format_search_output(task: str, result: ToolResult) -> str | None:
    if not result.success:
        return None
    payload = result.output if isinstance(result.output, dict) else {}
    if payload.get("mode") == "news" and isinstance(payload.get("articles"), list):
        clean = {
            "message": str(payload.get("message", "")).strip(),
            "insights": payload.get("insights", []),
            "articles": [
                {
                    "title": str(item.get("title", "")).strip(),
                    "description": str(item.get("description", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "source": str(item.get("source", "")).strip(),
                    "timestamp": str(item.get("timestamp", "")).strip(),
                }
                for item in payload.get("articles", [])
                if isinstance(item, dict)
            ],
        }
        return json.dumps(clean, ensure_ascii=False)

    items = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        return None

    lines = [f"Top results for: {task}"]
    seen_urls: set[str] = set()
    rendered_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "Untitled")).strip()
        snippet = str(item.get("snippet", "")).strip()
        url = str(item.get("url", "")).strip()
        normalized_url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE).rstrip("/").lower()
        if normalized_url and normalized_url in seen_urls:
            continue
        if normalized_url:
            seen_urls.add(normalized_url)

        rendered_count += 1
        if rendered_count > 3:
            break

        lines.append(title)
        if snippet:
            lines.append(f"   {snippet[:180]}")
        if url:
            lines.append(f"   Source: {url}")

    return "\n".join(lines)


def _run_heavy_shortcut(task: str) -> str | None:
    """Low-latency heavy path: run one targeted tool without full agent loop."""
    if _AGENT is None:
        return None

    text = (task or "").lower()
    web_search_needed = any(sig in text for sig in ("search", "latest", "news", "weather"))
    if not web_search_needed:
        return None

    tool = _AGENT.registry.get("web_search")
    if tool is None:
        return None

    result = tool.run(query=task)
    formatted = _format_search_output(task, result)
    if formatted:
        return formatted
    if result.success:
        return str(result.output)
    return None


def route_query(input_text: str) -> tuple[str, str | None]:
    """Classify user query to choose a low-latency execution path."""
    text = (input_text or "").strip().lower()
    if not text:
        return "fast", "How can I help you today?"

    if re.search(r"^(hi|hello|hey|yo|good\s*(morning|afternoon|evening))\b", text):
        return "fast", "Hello! How can I help you today?"

    if re.search(r"\b(thanks|thank\s*you|thx|appreciate\s*it)\b", text):
        return "fast", "You are welcome. Happy to help."

    if re.search(r"^help\b", text):
        return "fast", "I can answer questions, explain concepts, and run tool-backed tasks when needed. Tell me your goal."

    fast_patterns = [
        r"^(hi|hello|hey|yo|good\s*(morning|afternoon|evening))\b",
        r"\b(thanks|thank\s*you|thx|appreciate\s*it)\b",
        r"^help\b",
        r"\b(who are you|what can you do)\b",
    ]
    planner_signals = [
        "execution plan",
        "deadline",
        "priority",
        "time available",
        "time allocation",
        "workflow",
        "project manager",
        "plan this",
    ]
    # Tool-heavy path is intentionally strict to avoid unnecessary agent loops.
    heavy_signals = ["search", "latest", "news", "weather"]

    if any(re.search(p, text) for p in fast_patterns):
        return "fast", None
    if any(sig in text for sig in planner_signals):
        return "fast", None
    if any(sig in text for sig in heavy_signals):
        return "heavy", None

    # Most short general-knowledge prompts should use fast path.
    if len(text) <= 180:
        return "fast", None
    return "heavy", None


def build_agent() -> Agent:
    """
    Compose the system. Only place where components know about each other.
    Swap the model string in LLMClient() to use a different OpenRouter model.
    """
    # Runtime-configurable model/retries via environment variables.
    model = os.getenv("AGENT_MODEL", "qwen/qwen3.6-plus")
    llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
    llm = LLMClient(model=model, max_retries=llm_max_retries)

    registry = ToolRegistry()
    registry.register(QueryRefinerTool())
    registry.register(WebSearchTool())
    registry.register(CalculatorTool())
    registry.register(ConceptExplainerTool())
    registry.register(TextAnalyzerTool())
    registry.register(StudyPlanTool())

    return Agent(llm=llm, registry=registry)


def _build_memory_context(limit: int = 3) -> str:
    """Build prompt context from recent memories."""
    memories = get_recent_memories(limit)
    if not memories:
        return ""

    lines = ["Previous interactions:"]
    for idx, item in enumerate(reversed(memories), start=1):
        task_snippet = re.sub(r"\s+", " ", str(item.get("task", ""))).strip()[:120]
        result_snippet = re.sub(r"\s+", " ", str(item.get("result", ""))).strip()[:120]
        lines.append(f"{idx}. {task_snippet} -> {result_snippet}")
    return "\n".join(lines)


def initialize_runtime() -> Agent:
    """Initialize DB and singleton agent for both CLI and API entrypoints."""
    global _AGENT

    # Always initialize local DB on startup; do not rely on a pre-existing file.
    init_db()

    if not os.getenv("OPENROUTER_API_KEY"):
        if not os.getenv("CLAUDE_API_KEY"):
            raise RuntimeError(
                "OPENROUTER_API_KEY or CLAUDE_API_KEY environment variable not set. "
                "Get your key at: https://openrouter.ai/keys"
            )

    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


def run_agent(task: str) -> str:
    """
    Handles:
    - Claude/OpenRouter API call
    - reasoning / decision making
    - tool execution
    - returns final response
    """
    if _AGENT is None:
        raise RuntimeError("Agent is not initialized.")

    global _LAST_PLAN
    global _LAST_METRICS
    global _LAST_QUERY_NORMALIZED

    started_at = time.perf_counter()
    mode, instant_response = route_query(task)

    normalized_query = _normalize_query(task)
    is_news_query = _is_news_like_query(task)
    with _QUERY_STATE_LOCK:
        last_query = _LAST_QUERY_NORMALIZED
        similarity = _query_similarity(normalized_query, last_query or "") if last_query else 0.0
        intent_state = _classify_intent_state(task, last_query, similarity)
        is_repeat_query = intent_state == "repeat_request"
        # Track the latest query for repeat/similar-query checks.
        _LAST_QUERY_NORMALIZED = normalized_query or last_query

    avoid_cache = is_news_query and intent_state in {"follow_up", "repeat_request"}

    cached = None if avoid_cache else _cache_get(task)
    repeat_cache_hit = cached is not None
    if is_repeat_query and cached is None and last_query:
        cached = None if avoid_cache else _cache_get(last_query)
        repeat_cache_hit = cached is not None

    _RUNTIME_LOG.info(
        f"QUERY='{task}' normalized='{normalized_query}' "
        f"intent_state={intent_state} IS_REPEAT={is_repeat_query} "
        f"similarity={similarity:.2f} CACHE_HIT={repeat_cache_hit} avoid_cache={avoid_cache}"
    )

    if is_repeat_query and cached is not None and not is_news_query:
        refined = _build_repeat_response(task, cached)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        _LAST_PLAN = ""
        _LAST_METRICS = {
            "cache_hit": True,
            "mode": mode,
            "repeat_query": True,
            "intent_state": intent_state,
            "similarity": round(similarity, 3),
            "latency_ms": round(elapsed_ms, 2),
        }
        _cache_set(task, refined)
        _RUNTIME_LOG.info(f"mode={mode.upper()} repeat=1 cache=HIT latency_ms={elapsed_ms:.2f}")
        return refined

    if instant_response:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        _LAST_PLAN = ""
        _LAST_METRICS = {
            "mode": "fast",
            "cache_hit": False,
            "tool_calls": 0,
            "latency_ms": round(elapsed_ms, 2),
            "instant": True,
            "repeat_query": is_repeat_query,
            "intent_state": intent_state,
            "similarity": round(similarity, 3),
        }
        _cache_set(task, instant_response)
        _RUNTIME_LOG.info(f"mode=FAST cache=MISS instant=1 latency_ms={elapsed_ms:.2f}")
        return instant_response

    if cached is not None:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        _LAST_PLAN = ""
        _LAST_METRICS = {
            "cache_hit": True,
            "mode": mode,
            "latency_ms": round(elapsed_ms, 2),
            "repeat_query": is_repeat_query,
            "intent_state": intent_state,
            "similarity": round(similarity, 3),
        }
        _RUNTIME_LOG.info(f"mode={mode.upper()} cache=HIT latency_ms={elapsed_ms:.2f}")
        return cached

    memory_context = _build_memory_context(limit=1)
    task_with_context = task
    if memory_context:
        task_with_context = (
            f"{memory_context}\n\n"
            f"Current user task:\n{task}"
        )

    if mode == "fast":
        answer = _AGENT.quick_answer(task)
        _LAST_PLAN = ""
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        _LAST_METRICS = {
            "mode": "fast",
            "cache_hit": False,
            "tool_calls": 0,
            "latency_ms": round(elapsed_ms, 2),
            "repeat_query": is_repeat_query,
            "similarity": round(similarity, 3),
        }
        _cache_set(task, answer)
        _RUNTIME_LOG.info(f"mode=FAST cache=MISS latency_ms={elapsed_ms:.2f}")
        return answer

    heavy_shortcut = _run_heavy_shortcut(task)
    if heavy_shortcut:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        _LAST_PLAN = ""
        _LAST_METRICS = {
            "mode": "heavy",
            "cache_hit": False,
            "tool_calls": 1,
            "latency_ms": round(elapsed_ms, 2),
            "shortcut": True,
            "repeat_query": is_repeat_query,
            "intent_state": intent_state,
            "similarity": round(similarity, 3),
        }
        _cache_set(task, heavy_shortcut)
        _RUNTIME_LOG.info(f"mode=HEAVY shortcut=1 cache=MISS latency_ms={elapsed_ms:.2f}")
        return heavy_shortcut

    trace = _AGENT.run(task_with_context)
    _LAST_PLAN = trace.plan or ""
    _LAST_METRICS = trace.metrics or {}
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    _LAST_METRICS.update(
        {
            "mode": "heavy",
            "cache_hit": False,
            "latency_ms": round(elapsed_ms, 2),
            "repeat_query": is_repeat_query,
            "intent_state": intent_state,
            "similarity": round(similarity, 3),
        }
    )
    _RUNTIME_LOG.info(f"mode=HEAVY cache=MISS latency_ms={elapsed_ms:.2f}")
    if trace.success and trace.final_answer:
        _cache_set(task, trace.final_answer)
        return trace.final_answer
    return f"ABORTED: {trace.abort_reason or 'Unknown error'}"


def run_interactive():
    """Run an interactive terminal loop until the user exits."""
    print("AI Agent Started")
    print("Type 'exit' to quit.")

    while True:
        task = input("You: ").strip()
        if task.lower() == "exit":
            print("Exiting...")
            break
        if not task:
            continue

        result = run_agent(task)
        print(f"Agent: {result}\n")
        if _LAST_METRICS:
            print(
                "[Task Metrics] "
                f"tool_calls={_LAST_METRICS.get('tool_calls', 0)} "
                f"success={_LAST_METRICS.get('tool_success', 0)} "
                f"failures={_LAST_METRICS.get('tool_failures', 0)} "
                f"retries={_LAST_METRICS.get('llm_total_retries', 0)}"
            )

        memory_result = result
        if _LAST_PLAN:
            memory_result = f"PLAN:\n{_LAST_PLAN}\n\nRESULT:\n{result}"
        save_memory(task, memory_result)


def get_last_metrics() -> dict:
    """Expose latest runtime metrics for API logging."""
    return dict(_LAST_METRICS)


def main():
    try:
        initialize_runtime()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    run_interactive()


if __name__ == "__main__":
    main()
