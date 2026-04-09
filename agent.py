"""
Agent reasoning loop.

This is the heart of the system. The agent runs a ReAct loop:
  Observe → Think → Act → Observe → ... → Answer

It does not hard-code any decision logic. All decisions come
from the LLM. The loop's job is to faithfully execute those
decisions, handle failures gracefully, and know when to stop.
"""

import os
import re
from dataclasses import dataclass, field
from llm import LLMClient, LLMResponse
from tools import ToolRegistry, ToolResult
from logger import AgentLogger


# Safety cap: prevents runaway costs and infinite loops.
# A well-designed agent should finish most tasks in < 8 iterations.
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "2"))

_DEFAULT_ALLOWED_TOOLS = {
    "calculator",
    "web_search",
    "refine_query",
    "explain_concept",
    "analyze_text",
    "generate_study_plan",
}
_DANGEROUS_PATTERN = re.compile(
    r"(__|import\s+|exec\(|eval\(|subprocess|os\.|sys\.|open\(|rm\s+-rf|powershell|cmd\.exe|file://|ftp://)",
    flags=re.IGNORECASE,
)


@dataclass
class AgentTrace:
    """
    Full record of one agent run.
    Contains everything needed to explain the run in an interview.
    """
    task: str
    steps: list[dict] = field(default_factory=list)
    final_answer: str | None = None
    success: bool = False
    total_iterations: int = 0
    abort_reason: str | None = None
    plan: str | None = None
    metrics: dict = field(default_factory=dict)

    def add_step(self, iteration: int, thought: str, action: str,
                 tool_name: str | None = None, tool_result: str | None = None):
        self.steps.append({
            "iteration": iteration,
            "thought": thought,
            "action": action,
            "tool_name": tool_name,
            "tool_result": tool_result,
        })

    def summary(self) -> str:
        lines = [f"\n{'='*60}", f"TASK: {self.task}", f"{'='*60}"]
        if self.plan:
            lines.append("\nPLAN:")
            lines.append(self.plan)
        for step in self.steps:
            lines.append(f"\n[Step {step['iteration']}]")
            lines.append(f"  Thought : {step['thought'][:200]}")
            lines.append(f"  Action  : {step['action']}")
            if step["tool_name"]:
                lines.append(f"  Tool    : {step['tool_name']}")
            if step["tool_result"]:
                result_preview = step["tool_result"][:300]
                lines.append(f"  Result  : {result_preview}")
        lines.append(f"\n{'='*60}")
        if self.success:
            lines.append(f"ANSWER: {self.final_answer}")
        else:
            lines.append(f"ABORTED: {self.abort_reason}")
        lines.append(f"Total iterations: {self.total_iterations}")
        if self.metrics:
            lines.append(f"Metrics: {self.metrics}")
        lines.append("="*60)
        return "\n".join(lines)


class Agent:
    """
    Autonomous reasoning agent.

    Given a task, the agent iteratively reasons, selects tools,
    executes them, and incorporates results until it produces a
    final answer or hits a safety limit.

    The agent does not know what tools exist at init time — it
    discovers them from the registry at call time. This means
    tools can be hot-swapped without touching agent logic.
    """

    def __init__(self, llm: LLMClient, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry
        self.log = AgentLogger("Agent")
        self.enable_plan = os.getenv("AGENT_ENABLE_PLAN", "0") == "1"
        allowed = os.getenv("ALLOWED_TOOLS", "").strip()
        if allowed:
            self.allowed_tools = {t.strip() for t in allowed.split(",") if t.strip()}
        else:
            self.allowed_tools = set(_DEFAULT_ALLOWED_TOOLS)

    def _needs_tool(self, task: str) -> bool:
        text = (task or "").lower()
        if len(text) < 90 and not any(ch.isdigit() for ch in text):
            if any(w in text for w in ("hello", "hi", "thanks", "help", "who are you")):
                return False
        tool_signals = ("search", "latest", "news", "weather")
        return any(sig in text for sig in tool_signals)

    def quick_answer(self, task: str) -> str:
        return self.llm.quick_answer(task)

    def _contains_dangerous_pattern(self, value) -> bool:
        if isinstance(value, dict):
            return any(self._contains_dangerous_pattern(v) for v in value.values())
        if isinstance(value, list):
            return any(self._contains_dangerous_pattern(v) for v in value)
        if isinstance(value, str):
            return _DANGEROUS_PATTERN.search(value) is not None
        return False

    def _validate_tool_input(self, tool_name: str, tool_input: dict) -> str | None:
        if tool_name not in self.allowed_tools:
            return f"Tool '{tool_name}' is not allowed. Allowed tools: {sorted(self.allowed_tools)}"

        if self._contains_dangerous_pattern(tool_input):
            return f"Unsafe tool input detected for '{tool_name}'."

        if tool_name == "calculator":
            expr = str(tool_input.get("expression", ""))
            if len(expr) > 200:
                return "Calculator expression too long."
            if not re.fullmatch(r"[0-9+\-*/().,%^ a-zA-Z_]+", expr):
                return "Calculator expression contains unsafe characters."

        if tool_name == "web_search":
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return "Web search query cannot be empty."
            if len(query) > 300:
                return "Web search query too long."

        return None

    def _execute_tool_safely(self, tool_name: str, tool_input: dict) -> ToolResult:
        """Centralized tool execution guardrail layer."""
        validation_error = self._validate_tool_input(tool_name, tool_input)
        if validation_error:
            self.log.warn(validation_error)
            return ToolResult(success=False, error=validation_error)

        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found. Available: {self.registry.names()}",
            )

        try:
            return tool.run(**tool_input)
        except Exception as e:
            return ToolResult(success=False, error=f"Safe tool execution blocked crash: {e}")

    def _make_metrics(self, tool_calls: int, tool_success: int, tool_failures: int) -> dict:
        retry_stats = self.llm.retry_metrics() if hasattr(self.llm, "retry_metrics") else {}
        return {
            "tool_calls": tool_calls,
            "tool_success": tool_success,
            "tool_failures": tool_failures,
            "llm_total_retries": retry_stats.get("total_retry_events", 0),
            "llm_last_call_retries": retry_stats.get("last_call_retry_events", 0),
        }

    def _generate_plan(self, task: str) -> str:
        """Create a short execution plan before tool actions begin."""
        try:
            plan_prompt = (
                "Create a concise execution plan for this task in 2-5 numbered steps. "
                "Return action=final_answer and put only the plan text in answer.\n\n"
                f"Task: {task}"
            )
            plan_response = self.llm.call(
                messages=[{"role": "user", "content": plan_prompt}],
                tools=[],
            )
            if plan_response.action == "final_answer" and plan_response.answer:
                return plan_response.answer.strip()
        except RuntimeError as e:
            self.log.warn(f"Plan generation failed, using fallback plan: {e}")

        return "1. Identify required information\n2. Use appropriate tools\n3. Synthesize a clear final answer"

    def _best_effort_finalize(self, messages: list[dict], reason: str) -> str:
        """Try to force a final answer when progress stalls or tools fail repeatedly."""
        try:
            tail = messages[-4:] if len(messages) > 4 else messages
            final_prompt = (
                f"Reason: {reason}\n\n"
                "Use the context below to provide a concise final answer in plain text.\n"
                "Do not call tools. Keep answer under 120 words.\n\n"
                f"Context: {tail}"
            )
            return self.llm.quick_answer(final_prompt)
        except RuntimeError as e:
            self.log.warn(f"Best-effort finalization failed: {e}")

        return (
            "I could not complete every step with external tools, but based on available context, "
            "I provided the best possible answer path. Please refine the query with specific entities "
            "or timeframe for a stronger result."
        )

    def run(self, task: str) -> AgentTrace:
        """
        Execute the agent loop for a given task.
        Returns a full AgentTrace regardless of success or failure.
        """
        self.log.info(f"Starting task: {task}")
        trace = AgentTrace(task=task)

        if not self._needs_tool(task):
            try:
                trace.final_answer = self.quick_answer(task)
                trace.success = True
                trace.total_iterations = 1
                trace.metrics = self._make_metrics(0, 0, 0)
                trace.add_step(
                    iteration=1,
                    thought="Task classified as no-tool query; used fast direct completion.",
                    action="final_answer",
                )
                return trace
            except RuntimeError as e:
                self.log.warn(f"Fast no-tool path failed, falling back to heavy loop: {e}")

        # Conversation history. We build this incrementally.
        # The LLM has no memory — we carry it.
        messages = [{"role": "user", "content": task}]

        if self.enable_plan:
            plan_text = self._generate_plan(task)
            trace.plan = plan_text
            trace.add_step(iteration=0, thought="Generated execution plan", action="plan", tool_result=plan_text)
            messages.append({
                "role": "user",
                "content": (
                    "Execution plan for this task:\n"
                    f"{plan_text}\n\n"
                    "Follow this plan and adapt if tools fail."
                ),
            })

        # Track last 3 tool calls to detect loops.
        # If the agent repeats the same call 3 times, something is wrong.
        recent_calls: list[str] = []
        failed_call_counts: dict[str, int] = {}
        total_tool_failures = 0
        total_tool_calls = 0
        successful_tool_calls = 0

        for iteration in range(1, MAX_ITERATIONS + 1):
            self.log.info(f"--- Iteration {iteration}/{MAX_ITERATIONS} ---")

            # Get the model's next reasoning step
            try:
                scoped_messages = messages[-6:] if len(messages) > 6 else messages
                response: LLMResponse = self.llm.call(
                    messages=scoped_messages,
                    tools=self.registry.all_schemas(),
                )
            except RuntimeError as e:
                # LLM call failed after all retries
                trace.final_answer = self._best_effort_finalize(messages, reason=f"LLM failure: {e}")
                trace.success = True
                trace.total_iterations = iteration
                trace.metrics = self._make_metrics(total_tool_calls, successful_tool_calls, total_tool_failures)
                self.log.error(f"LLM call failed, returned best-effort answer: {e}")
                return trace

            self.log.thought(response.thought)

            # Agent is done — extract final answer
            if response.action == "final_answer":
                if not response.answer or not str(response.answer).strip():
                    trace.abort_reason = "Model returned final_answer action with empty answer."
                    trace.total_iterations = iteration
                    self.log.error(trace.abort_reason)
                    return trace
                trace.add_step(
                    iteration=iteration,
                    thought=response.thought,
                    action="final_answer",
                )
                trace.final_answer = response.answer
                trace.success = True
                trace.total_iterations = iteration
                trace.metrics = self._make_metrics(total_tool_calls, successful_tool_calls, total_tool_failures)
                self.log.info(f"Agent reached final answer in {iteration} iterations.")
                return trace

            # Agent wants to call a tool
            if response.action == "tool_call":
                if not self._needs_tool(task):
                    trace.final_answer = self._best_effort_finalize(
                        messages,
                        reason="Tool call skipped by latency policy for non-tool query",
                    )
                    trace.success = True
                    trace.total_iterations = iteration
                    trace.metrics = self._make_metrics(total_tool_calls, successful_tool_calls, total_tool_failures)
                    return trace

                if total_tool_calls >= 1:
                    trace.final_answer = self._best_effort_finalize(
                        messages,
                        reason="Tool-call cap reached for latency budget",
                    )
                    trace.success = True
                    trace.total_iterations = iteration
                    trace.metrics = self._make_metrics(total_tool_calls, successful_tool_calls, total_tool_failures)
                    return trace

                tool_name = response.tool_name
                tool_input = response.tool_input or {}

                # Guard: check the tool actually exists
                tool = self.registry.get(tool_name)
                if tool is None:
                    error_msg = f"Tool '{tool_name}' not found. Available: {self.registry.names()}"
                    self.log.warn(error_msg)
                    result_text = f"TOOL_ERROR: {error_msg}"
                else:
                    # Detect loop: same tool+input called 3 times in a row
                    call_sig = f"{tool_name}:{tool_input}"
                    recent_calls.append(call_sig)
                    if len(recent_calls) > 3:
                        recent_calls.pop(0)

                    if recent_calls.count(call_sig) >= 3:
                        trace.abort_reason = (
                            f"Loop detected: '{tool_name}' called with same "
                            f"inputs {recent_calls.count(call_sig)} times."
                        )
                        trace.total_iterations = iteration
                        self.log.error(trace.abort_reason)
                        return trace

                    self.log.tool_call(tool_name, tool_input)
                    total_tool_calls += 1

                    # Execute the tool — tools handle their own errors internally
                    result: ToolResult = self._execute_tool_safely(tool_name, tool_input)
                    result_text = result.as_message()

                    if not result.success:
                        self.log.warn(f"Tool '{tool_name}' failed: {result.error}")
                        total_tool_failures += 1
                        failed_call_counts[call_sig] = failed_call_counts.get(call_sig, 0) + 1

                        if failed_call_counts[call_sig] >= 2:
                            self.log.warn(
                                f"Repeated tool failure detected for {tool_name}. "
                                "Nudging model to rewrite query or choose another tool."
                            )
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Tool {tool_name} failed repeatedly with the same input. "
                                    "Do not repeat this exact call. "
                                    "Try rewriting inputs or selecting a different tool."
                                ),
                            })

                        if total_tool_failures >= 4:
                            trace.final_answer = self._best_effort_finalize(
                                messages,
                                reason="Excessive tool failures across the task",
                            )
                            trace.success = True
                            trace.total_iterations = iteration
                            trace.metrics = self._make_metrics(
                                total_tool_calls,
                                successful_tool_calls,
                                total_tool_failures,
                            )
                            self.log.error("Excessive tool failures, returned best-effort answer.")
                            return trace
                    else:
                        successful_tool_calls += 1
                        self.log.tool_result(tool_name, result_text[:200])

                trace.add_step(
                    iteration=iteration,
                    thought=response.thought,
                    action="tool_call",
                    tool_name=tool_name,
                    tool_result=result_text,
                )

                # Append this turn to conversation history so the model
                # can reason about what it just learned
                messages.append({
                    "role": "assistant",
                    "content": response.raw,
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result from {tool_name}:\n{result_text}\n\n"
                        "Use this result. Call another tool only if strictly needed; otherwise return final_answer."
                    ),
                })

        # Reached MAX_ITERATIONS without a final answer
        trace.final_answer = self._best_effort_finalize(
            messages,
            reason=f"Reached maximum iterations ({MAX_ITERATIONS})",
        )
        trace.success = True
        trace.total_iterations = MAX_ITERATIONS
        trace.metrics = self._make_metrics(total_tool_calls, successful_tool_calls, total_tool_failures)
        self.log.error(f"Reached maximum iterations ({MAX_ITERATIONS}), returned best-effort answer.")
        return trace
