"""Structured agent records, independent of datasets and memory backends.

These records describe source activity, not a schedule of LTM requests. A
coding-agent scenario can decide which activity causes an add or search.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from ltm100.common import UserId


@dataclass(frozen=True)
class AgentToolCall:
    """One recorded tool invocation; arguments are data, never executable."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentMessage:
    """One source message, retaining call/observation links and ordering."""

    role: str
    content: str
    tool_calls: tuple[AgentToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class AgentTask:
    """One independent execution attempt, including interrupted attempts.

    ``resolved`` describes the source coding task, not LTM request success.
    ``final_response`` is absent when the source has no terminal response.
    """

    trajectory_id: str
    instance_id: str
    repo: str
    messages: tuple[AgentMessage, ...]
    exit_status: str
    resolved: bool
    final_response: str | None = None


class AgentDataset(Protocol):
    """Optional dataset capability for finite, structured agent task streams.

    Iteration yields a user's assigned tasks once. Repetition, task gaps, and
    selective LTM calls belong to the consuming scenario, not the dataset.
    """

    def task_stream(self, user: UserId) -> Iterator[AgentTask]: ...


__all__ = ["AgentDataset", "AgentMessage", "AgentTask", "AgentToolCall"]
