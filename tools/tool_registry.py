from __future__ import annotations

from typing import Any


class ToolRegistry:
    """One shared registry for research tool instances used by all agents."""

    def __init__(self, **tools: Any):
        self._tools: dict[str, Any] = {}
        for name, tool in tools.items():
            if tool is not None:
                self.register(name, tool)

    def register(self, name: str, tool: Any) -> None:
        normalized = name.strip().casefold()
        if not normalized or not normalized.replace("_", "").isalnum():
            raise ValueError("Tool names may contain only letters, numbers, and underscores.")
        if normalized in self._tools:
            raise ValueError(f"Tool '{normalized}' is already registered.")
        self._tools[normalized] = tool

    def get(self, name: str) -> Any:
        try:
            return self._tools[name.casefold()]
        except KeyError as exc:
            raise KeyError(f"Research tool '{name}' is not available.") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))
