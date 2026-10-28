from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from observability import log_event


class ActionLog:
    """Small live action list whose renderer can target a Streamlit placeholder."""

    def __init__(self, render: Callable[[list[dict[str, str]]], Any]):
        self._render = render
        self._entries: list[dict[str, str]] = []

    def emit(self, action: str, status: str = "complete", detail: str = "") -> None:
        action = " ".join(str(action).split())[:180] or "Agent step"
        detail = str(detail)[:800]
        status = status if status in {"running", "complete", "failed", "terminated"} else "complete"
        entry = next((row for row in reversed(self._entries) if row["action"] == action and row["status"] == "running"), None)
        if entry is None:
            entry = {"action": action, "status": status, "detail": detail}
            self._entries.append(entry)
        else:
            entry["status"] = status
            entry["detail"] = detail
        action_label = "Researching task" if action.startswith("Researching:") else action
        log_event("agent_action", tool=action_label, status=status)
        self._publish()

    def snapshot(self) -> list[dict[str, str]]:
        return [dict(entry) for entry in self._entries]

    def terminate(self, detail: str = "Stopped before completion") -> None:
        running = [entry for entry in self._entries if entry["status"] == "running"]
        if running:
            for entry in running:
                entry["status"] = "terminated"
                entry["detail"] = detail
        else:
            self._entries.append({
                "action": "Request terminated",
                "status": "terminated",
                "detail": detail,
            })

    def _publish(self) -> None:
        self._render(self.snapshot())


def render_saved_actions(st, actions: list[dict[str, str]]) -> None:
    if not actions:
        return
    with st.expander("Agent actions", expanded=False):
        for index, entry in enumerate(actions, start=1):
            status = entry.get("status", "complete")
            icon = "✓" if status == "complete" else "!" if status == "failed" else "■" if status == "terminated" else "◌"
            label = entry.get("action", "Action")
            with st.expander(f"{index}. {icon} {label}", expanded=False):
                state = ("In progress" if status == "running" else "Completed" if status == "complete"
                         else "Terminated" if status == "terminated" else "Failed")
                st.caption(f"Status: {state}")
                detail = entry.get("detail", "")
                st.text(detail or "No additional details for this step.")


def render_live_actions(placeholder, actions: list[dict[str, str]]) -> None:
    placeholder.empty()
    with placeholder.container():
        for index, entry in enumerate(actions, start=1):
            status = entry.get("status", "complete")
            icon = "✓" if status == "complete" else "!" if status == "failed" else "■" if status == "terminated" else "◌"
            label = entry.get("action", "Action")
            with st.expander(f"{index}. {icon} {label}", expanded=False):
                state = ("In progress" if status == "running" else "Completed" if status == "complete"
                         else "Terminated" if status == "terminated" else "Failed")
                st.caption(f"Status: {state}")
                detail = entry.get("detail", "")
                st.text(detail or "No additional details for this step yet.")
