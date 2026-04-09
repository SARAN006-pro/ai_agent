"""
LLM abstraction layer — OpenRouter backend.

OpenRouter exposes an OpenAI-compatible API, so we use the
official `openai` Python SDK pointed at https://openrouter.ai/api/v1.

Default model: anthropic/claude-opus-4-5
Change DEFAULT_MODEL to switch to any model on OpenRouter with
zero changes elsewhere in the codebase.
"""

import os
import time
import json
import random
import re
from dataclasses import dataclass
from logger import AgentLogger

try:
    from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
except ImportError:
    OpenAI = None

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Switch models here — nothing else needs to change.
# Other options: "anthropic/claude-sonnet-4-5", "openai/gpt-4o",
#                "meta-llama/llama-3.1-70b-instruct"
DEFAULT_MODEL = "qwen/qwen3.6-plus"

FAST_SYSTEM_PROMPT = (
    "You are a fast helpful assistant. "
    "Answer directly and concisely using plain text. "
    "Do not call tools or produce JSON. "
    "IMPORTANT RULE: Always complete your response fully. "
    "Do NOT cut off mid-sentence. Do NOT stop in the middle of a list. "
    "Ensure all steps are completed before ending."
)

PLANNER_SYSTEM_PROMPT = (
    "You are an execution-focused AI planner. "
    "Create a time-bound execution plan, not a summary. "
    "Adapt to user constraints: task, deadline, available time, and priority.\n\n"
    "Rules:\n"
    "- Do not repeat the input.\n"
    "- Do not give generic long lists.\n"
    "- Prioritize only top 4-6 actions.\n"
    "- Assign time to each step (minutes/hours).\n"
    "- Clearly state what must be done first.\n"
    "- Include what to skip under deadline pressure.\n"
    "- Optimize for completion, not perfection.\n\n"
    "Use this exact format:\n"
    "Execution Plan\n"
    "Time Allocation\n"
    "Priority Order\n"
    "Workflow (First -> Last)\n"
    "What To Skip\n\n"
    "IMPORTANT RULE: Always complete your response fully. "
    "Do NOT cut off mid-sentence or mid-list."
)

SYSTEM_PROMPT = """You are an autonomous agent.

On every turn, you MUST respond in this exact JSON format:
{
  "thought": "Your internal reasoning about the current state and what to do next",
  "action": "tool_call" | "final_answer",
  "tool_name": "<name of tool to call, only when action=tool_call>",
  "tool_input": {<tool arguments as object, only when action=tool_call>},
  "answer": "<final answer text, only when action=final_answer>"
}

Rules:
- Call tools only when strictly necessary.
- If task can be answered from general knowledge, return final_answer directly.
- Do not repeat previously shown results when the query is similar to recent turns.
- For follow-up or repeated requests, prioritize novel, diverse, and unexplored angles.
- If similar-query context is provided, avoid reusing the same links unless no alternatives exist.
- Do NOT wrap the JSON in markdown backticks.
- IMPORTANT RULE: Always complete your response fully.
- Do NOT cut off mid-sentence.
- Do NOT stop in the middle of a list.
- Ensure all steps are completed before ending.
"""


@dataclass
class LLMResponse:
    """Parsed response from one agent turn."""
    thought: str
    action: str
    tool_name: str | None
    tool_input: dict | None
    answer: str | None
    raw: str


class LLMClient:
    """
    Wraps the OpenRouter API (OpenAI-compatible) with retry logic,
    structured output parsing, and clean error surfacing.

    We use JSON-in-system-prompt rather than native tool calling
    because not all OpenRouter models support function calling.
    This keeps the agent logic fully model-agnostic.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 2):
        if OpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY or CLAUDE_API_KEY environment variable not set.\n"
                "Get your key at: https://openrouter.ai/keys"
            )

        self.client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self.model = model
        self.fast_model = os.getenv("FAST_LLM_MODEL", "openai/gpt-4o-mini")
        configured_retries = int(os.getenv("LLM_MAX_RETRIES", str(max_retries)))
        self.max_retries = min(configured_retries, 4)
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "500"))
        self.fast_max_tokens = int(os.getenv("FAST_LLM_MAX_TOKENS", "320"))
        self.continuation_max_tokens = int(os.getenv("LLM_CONTINUATION_MAX_TOKENS", "220"))
        self.max_continuation_rounds = int(os.getenv("LLM_MAX_CONTINUATION_ROUNDS", "2"))
        self.backoff_base_seconds = float(os.getenv("LLM_BACKOFF_BASE_SECONDS", "0.2"))
        self.backoff_cap_seconds = float(os.getenv("LLM_BACKOFF_CAP_SECONDS", "2.0"))
        self.total_retry_events = 0
        self.last_call_retry_events = 0
        self.log = AgentLogger("LLMClient")
        self.log.info(
            f"Using model: {self.model} via OpenRouter "
            f"(max_retries={self.max_retries}, max_tokens={self.max_tokens})"
        )

    def _is_incomplete_text(self, text: str | None) -> bool:
        """Heuristic guard to detect clearly cut-off model output."""
        t = (text or "").rstrip()
        if not t:
            return True

        planner_headings = [
            "execution plan",
            "time allocation",
            "priority order",
            "workflow (first -> last)",
            "what to skip",
        ]
        lower_t = t.lower()
        if all(h in lower_t for h in planner_headings):
            return False

        if t.endswith(":") or t.endswith("-") or t.endswith("..."):
            return True

        if re.search(r"\n\s*(?:[-*]|\d+\.)\s*$", t):
            return True

        # Long responses ending in a bare word are often truncated by token limits.
        if len(t) > 80 and re.search(r"[A-Za-z0-9]$", t) and not re.search(r"[.!?\]\)\"]$", t):
            return True

        unbalanced_pairs = [("(", ")"), ("[", "]"), ("{", "}"), ('"', '"')]
        for left, right in unbalanced_pairs:
            if t.count(left) > t.count(right):
                return True

        return False

    def _merge_continuation(self, base: str, continuation: str) -> str:
        left = (base or "").rstrip()
        right = (continuation or "").lstrip()
        if not left:
            return right
        if not right:
            return left

        # Ignore pure repetition from continuation calls.
        if right in left:
            return left

        # If continuation starts by repeating a known heading already present, trim it.
        if "Execution Plan" in left and right.startswith("Execution Plan"):
            first_break = right.find("\n")
            if first_break != -1:
                right = right[first_break + 1 :].lstrip()
                if not right:
                    return left

        # Merge with suffix/prefix overlap removal to avoid duplicated blocks.
        max_overlap = min(len(left), len(right), 300)
        overlap_len = 0
        for size in range(max_overlap, 19, -1):
            if left[-size:] == right[:size]:
                overlap_len = size
                break
        if overlap_len > 0:
            return left + right[overlap_len:]

        if left.endswith(("\n", " ")):
            return left + right
        return left + " " + right

    def _continue_plain_response(self, messages: list[dict], current_text: str, model_name: str) -> str:
        """Request continuation chunks until output no longer looks truncated."""
        merged = (current_text or "").strip()
        if not self._is_incomplete_text(merged):
            return merged

        for _ in range(self.max_continuation_rounds):
            continuation_messages = messages + [
                {"role": "assistant", "content": merged},
                {
                    "role": "user",
                    "content": (
                        "Continue the previous response properly from where it stopped. "
                        "Do not restart, do not repeat, and ensure the structure is fully completed."
                    ),
                },
            ]
            continuation = self.client.chat.completions.create(
                model=model_name,
                max_tokens=self.continuation_max_tokens,
                messages=continuation_messages,
            )
            cont_text = continuation.choices[0].message.content if continuation.choices else ""
            merged = self._merge_continuation(merged, str(cont_text or ""))
            if not self._is_incomplete_text(merged):
                break

        return merged.strip()

    def _continue_structured_response(self, full_messages: list[dict], parsed: LLMResponse) -> LLMResponse:
        """Continue a truncated final_answer while preserving JSON contract."""
        merged_answer = (parsed.answer or "").strip()
        if parsed.action != "final_answer" or not self._is_incomplete_text(merged_answer):
            return parsed

        for _ in range(self.max_continuation_rounds):
            continuation_messages = full_messages + [
                {"role": "assistant", "content": parsed.raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous final answer appears incomplete. Continue it from where it ended. "
                        "Respond ONLY with valid JSON in the same schema. "
                        "Set action to final_answer and put ONLY continuation text in answer."
                    ),
                },
            ]
            continuation = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.continuation_max_tokens,
                messages=continuation_messages,
            )
            raw_cont = continuation.choices[0].message.content if continuation.choices else ""
            cont_parsed = self._parse(str(raw_cont or ""))

            if cont_parsed.action != "final_answer":
                break

            merged_answer = self._merge_continuation(merged_answer, cont_parsed.answer or "")
            parsed = LLMResponse(
                thought=cont_parsed.thought or parsed.thought,
                action="final_answer",
                tool_name=None,
                tool_input=None,
                answer=merged_answer,
                raw=parsed.raw,
            )
            if not self._is_incomplete_text(merged_answer):
                break

        return parsed

    def _is_planning_request(self, prompt: str) -> bool:
        text = (prompt or "").lower()
        planner_signals = (
            "execution plan",
            "deadline",
            "priority",
            "time available",
            "time allocation",
            "workflow",
            "task planning",
            "project manager",
            "plan this",
        )
        return any(signal in text for signal in planner_signals)

    def _parse_json_object(self, raw_text: str) -> dict:
        text = (raw_text or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.startswith("```"))
            text = text.strip()

        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Vision response is not a JSON object.")
        return data

    def vision_analyze(self, image_data_url: str) -> dict:
        """Analyze an image using a vision-capable model and return structured JSON."""
        model_name = os.getenv("VISION_LLM_MODEL", "openai/gpt-4o-mini")
        prompt = (
            "Analyze this image and return STRICT JSON only with this schema:\n"
            "{\n"
            '  "objects": ["..."],\n'
            '  "text": "",\n'
            '  "description": "",\n'
            '  "possible_actions": ["..."]\n'
            "}\n"
            "Rules: no markdown, no explanations, ensure valid JSON."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    max_tokens=self.max_tokens,
                    messages=messages,
                )
                if not getattr(response, "choices", None):
                    raise RuntimeError("Vision model returned no choices.")

                raw = response.choices[0].message.content or ""
                data = self._parse_json_object(str(raw))
                return {
                    "objects": data.get("objects", []) if isinstance(data.get("objects"), list) else [],
                    "text": str(data.get("text", "") or ""),
                    "description": str(data.get("description", "") or ""),
                    "possible_actions": (
                        data.get("possible_actions", [])
                        if isinstance(data.get("possible_actions"), list)
                        else []
                    ),
                }
            except Exception as e:
                last_error = e
                wait = self._backoff_delay(attempt)
                time.sleep(wait)

        raise RuntimeError(f"Vision analysis failed: {last_error}")

    def quick_answer(self, prompt: str) -> str:
        """Low-latency text completion path for simple queries (no tools, no JSON)."""
        is_planner_mode = self._is_planning_request(prompt)
        system_prompt = PLANNER_SYSTEM_PROMPT if is_planner_mode else FAST_SYSTEM_PROMPT
        max_tokens = self.max_tokens if is_planner_mode else self.fast_max_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Try fast model first; fall back to main model if unavailable.
        for model_name in (self.fast_model, self.model):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                if not getattr(response, "choices", None):
                    continue
                text = response.choices[0].message.content
                if text and str(text).strip():
                    completed = self._continue_plain_response(messages, str(text).strip(), model_name)
                    return completed
            except Exception as e:
                self.log.warn(f"quick_answer failed on model={model_name}: {e}")

        raise RuntimeError("quick_answer failed on both fast and primary models.")

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter and hard cap."""
        exp = self.backoff_base_seconds * (2 ** attempt)
        jitter = random.uniform(0, 0.35)
        return min(exp + jitter, self.backoff_cap_seconds)

    def call(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """
        Send messages to the model and return a parsed LLMResponse.
        Tool schemas are injected into the system prompt as text so
        this works across all OpenRouter models uniformly.
        Retries on transient errors with exponential backoff.
        """
        last_error = None
        self.last_call_retry_events = 0
        tool_desc = self._format_tools_for_prompt(tools)

        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tool_desc}
        ] + messages

        last_signature = ""
        repeated_signature_count = 0

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=full_messages,
                )
                if not getattr(response, "choices", None):
                    raise RuntimeError(f"Model returned no choices: {response}")
                raw_text = response.choices[0].message.content
                if raw_text is None or str(raw_text).strip() == "":
                    raise RuntimeError("Model returned empty content.")
                parsed = self._parse(raw_text)
                parsed = self._continue_structured_response(full_messages, parsed)
                return parsed

            except RateLimitError as e:
                wait = self._backoff_delay(attempt)
                self.log.warn(f"Rate limited. Waiting {wait:.2f}s (attempt {attempt+1})")
                time.sleep(wait)
                last_error = e
                self.last_call_retry_events += 1
                self.total_retry_events += 1

            except APIConnectionError as e:
                wait = self._backoff_delay(attempt)
                self.log.warn(f"Connection error on attempt {attempt+1}: {e}. Waiting {wait:.2f}s")
                time.sleep(wait)
                last_error = e
                self.last_call_retry_events += 1
                self.total_retry_events += 1

            except APIStatusError as e:
                wait = self._backoff_delay(attempt)
                self.log.warn(
                    f"API error {e.status_code} on attempt {attempt+1}: {e.message}. "
                    f"Waiting {wait:.2f}s"
                )
                time.sleep(wait)
                last_error = e
                self.last_call_retry_events += 1
                self.total_retry_events += 1

            except ValueError as e:
                # Malformed JSON — inject correction nudge and retry
                self.log.warn(f"Parse failure on attempt {attempt+1}: {e}")
                full_messages = full_messages + [{
                    "role": "user",
                    "content": "Your last response was not valid JSON. Respond ONLY with the JSON object, no markdown, no backticks."
                }]
                wait = self._backoff_delay(attempt)
                time.sleep(wait)
                last_error = e
                self.last_call_retry_events += 1
                self.total_retry_events += 1

            except RuntimeError as e:
                wait = self._backoff_delay(attempt)
                self.log.warn(f"Runtime response issue on attempt {attempt+1}: {e}. Waiting {wait:.2f}s")
                time.sleep(wait)
                last_error = e
                self.last_call_retry_events += 1
                self.total_retry_events += 1

            signature = f"{type(last_error).__name__}:{str(last_error)[:160]}"
            if signature == last_signature:
                repeated_signature_count += 1
            else:
                repeated_signature_count = 1
                last_signature = signature

            if repeated_signature_count >= 2 and attempt < self.max_retries - 1:
                self.log.warn("Repeated identical LLM failure detected; aborting retries early.")
                break

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def retry_metrics(self) -> dict:
        """Expose lightweight retry metrics for task-level summaries."""
        return {
            "last_call_retry_events": self.last_call_retry_events,
            "total_retry_events": self.total_retry_events,
        }

    def _format_tools_for_prompt(self, tools: list[dict]) -> str:
        """Convert Anthropic-format tool schemas into readable text for the system prompt."""
        if not tools:
            return ""
        lines = ["AVAILABLE TOOLS:", ""]
        for tool in tools:
            props = tool.get("input_schema", {}).get("properties", {})
            required = tool.get("input_schema", {}).get("required", [])
            lines.append(f"Tool: {tool['name']}")
            lines.append(f"  Description: {tool['description']}")
            if props:
                lines.append("  Parameters:")
                for pname, pdef in props.items():
                    req = " (required)" if pname in required else " (optional)"
                    lines.append(f"    - {pname}: {pdef.get('description', pdef.get('type', ''))}{req}")
            lines.append("")
        return "\n".join(lines)

    def _parse(self, raw: str) -> LLMResponse:
        """
        Parse structured JSON from model output.
        Strips markdown fences if present. Raises ValueError on bad
        structure so the caller can retry with a correction nudge.
        """
        text = (raw or "").strip()
        if not text:
            raise ValueError("Model returned empty response.")

        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()

        # Some models prepend/append text around JSON; extract the first JSON object.
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from model: {e}\nRaw: {raw[:300]}")

        missing = {"thought", "action"} - data.keys()
        if missing:
            raise ValueError(f"Model response missing fields: {missing}")

        action = data.get("action")
        if action not in ("tool_call", "final_answer"):
            raise ValueError(f"Unknown action: {action!r}")

        return LLMResponse(
            thought=data.get("thought", ""),
            action=action,
            tool_name=data.get("tool_name"),
            tool_input=data.get("tool_input", {}),
            answer=data.get("answer"),
            raw=raw,
        )
