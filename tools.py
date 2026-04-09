"""
Tool abstraction and registry.

Each tool is a self-contained unit: it knows its own schema,
how to run itself, and how to handle its own errors.
The registry is the single source of truth the agent uses
to discover and call tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """
    Uniform result wrapper from any tool execution.
    success=False means the agent should reason about the failure,
    not crash. The error field gives context for that reasoning.
    """
    success: bool
    output: Any = None   # structured data or string
    error: str | None = None

    def as_message(self) -> str:
        """Human-readable form for injecting into the conversation."""
        if self.success:
            return str(self.output)
        return f"TOOL_ERROR: {self.error}"


class BaseTool(ABC):
    """
    Every tool must implement name, description, schema, and run.
    This contract is what the registry enforces.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used by the agent to call this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Plain-English description of what the tool does and when to use it."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict:
        """JSON Schema for this tool's inputs."""
        ...

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool. Must never raise — catch and return ToolResult(success=False)."""
        ...

    def to_anthropic_schema(self) -> dict:
        """Format this tool for the Claude API tools parameter."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """
    Centralized store for all available tools.
    The agent queries this to know what it can do.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all_schemas(self) -> list[dict]:
        """Return schemas for all registered tools, for the API call."""
        return [t.to_anthropic_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())
