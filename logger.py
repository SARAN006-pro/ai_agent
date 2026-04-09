"""
Structured logger for agent observability.

In a real system this would write structured JSON to a log sink.
Here it writes to stdout with clear visual hierarchy so you can
follow the agent's reasoning in real time during an interview demo.
"""

import sys
from datetime import datetime


class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GRAY    = "\033[90m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"


class AgentLogger:
    """
    Per-component logger. Prefix identifies which part of the
    system produced each log line — critical for debugging
    multi-component agents.
    """

    def __init__(self, component: str):
        self.component = component

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _write(self, color: str, level: str, msg: str):
        line = (
            f"{Colors.GRAY}{self._ts()}{Colors.RESET} "
            f"{color}{Colors.BOLD}[{level}]{Colors.RESET} "
            f"{Colors.GRAY}[{self.component}]{Colors.RESET} "
            f"{msg}"
        )
        print(line, flush=True)

    def info(self, msg: str):
        self._write(Colors.CYAN, "INFO ", msg)

    def warn(self, msg: str):
        self._write(Colors.YELLOW, "WARN ", msg)

    def error(self, msg: str):
        self._write(Colors.RED, "ERROR", msg)

    def thought(self, msg: str):
        """Agent's internal reasoning — visually distinct."""
        print(
            f"\n{Colors.MAGENTA}{Colors.BOLD}  ◆ THOUGHT{Colors.RESET} "
            f"{Colors.MAGENTA}{msg}{Colors.RESET}\n",
            flush=True,
        )

    def tool_call(self, tool: str, inputs: dict):
        """Log a tool invocation with its arguments."""
        args = ", ".join(f"{k}={repr(v)}" for k, v in inputs.items())
        self._write(Colors.BLUE, "TOOL▶", f"{Colors.BOLD}{tool}({args}){Colors.RESET}")

    def tool_result(self, tool: str, result: str):
        """Log what a tool returned (truncated for readability)."""
        preview = result.replace("\n", " ")[:180]
        self._write(Colors.GREEN, "TOOL◀", f"{tool} → {preview}")
