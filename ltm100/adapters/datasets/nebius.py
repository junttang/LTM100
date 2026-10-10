"""Nebius SWE-rebench OpenHands trajectories for agent and corpus workloads.

The adapter retains source events without executing tools or choosing LTM
calls. Hugging Face/Parquet data stays disk-backed; records are normalized on
access rather than materializing the whole dataset as Python objects.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ltm100.agent import AgentMessage, AgentTask, AgentToolCall

DATASET_ID = "nebius/SWE-rebench-openhands-trajectories"
DEFAULT_REVISION = "35455389ab51bf5e2306bfd436ef72d0f98bf882"


class NebiusAdapter:
    """Load independent coding-agent execution attempts, including failures."""

    name = "nebius"

    def __init__(
        self,
        length: int | None = None,
        cache_dir: str | None = None,
        path: str | None = None,
        revision: str = DEFAULT_REVISION,
        chunk_chars: int = 3000,
    ) -> None:
        if length is not None and (
            isinstance(length, bool) or not isinstance(length, int) or length < 0
        ):
            raise ValueError("nebius length must be a non-negative integer or None")
        if (
            isinstance(chunk_chars, bool)
            or not isinstance(chunk_chars, int)
            or chunk_chars <= 0
        ):
            raise ValueError("nebius chunk_chars must be a positive integer")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("nebius revision must be a non-empty string")
        self.length = length
        self.cache_dir = cache_dir
        self.path = path
        self.revision = revision
        self.chunk_chars = chunk_chars
        self._records: Sequence[dict[str, Any]] | None = None

    # -- loading -----------------------------------------------------------

    def _load(self) -> Sequence[dict[str, Any]]:
        if self._records is not None:
            return self._records
        if self.path:
            source = Path(self.path).expanduser()
            if source.suffix.lower() == ".parquet":
                self._records = self._load_arrow(source)
            else:
                self._records = self._load_json(source)
        else:
            self._records = self._load_arrow()
        return self._records

    def _load_arrow(self, source: Path | None = None) -> Sequence[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                'Nebius Hugging Face/Parquet loading requires pip install ".[datasets]"'
            ) from exc

        if source is None:
            records = load_dataset(
                DATASET_ID,
                split="train",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            records = load_dataset(
                "parquet",
                data_files=str(source),
                split="train",
                cache_dir=self.cache_dir,
            )
        if "trajectory" not in records.column_names:
            raise ValueError("nebius dataset is missing the trajectory column")
        if self.length is not None:
            records = records.select(range(min(self.length, len(records))))
        return records

    def _load_json(self, source: Path) -> list[dict[str, Any]]:
        if source.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("nebius path must be a .json, .jsonl, or .parquet file")
        records: list[dict[str, Any]] = []
        with source.open("rb") as stream:
            if source.suffix.lower() == ".jsonl":
                for line_number, line in enumerate(stream, 1):
                    if self.length is not None and len(records) >= self.length:
                        break
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError as exc:
                        raise ValueError(
                            f"invalid nebius JSON at {source}:{line_number}"
                        ) from exc
                    records.append(record)
            else:
                import ijson

                try:
                    first = next(ijson.parse(stream), None)
                    if first is None or first[1] != "start_array":
                        raise ValueError(f"nebius JSON must contain an array: {source}")
                    stream.seek(0)
                    items = ijson.items(stream, "item", use_float=True)
                    while self.length is None or len(records) < self.length:
                        try:
                            records.append(next(items))
                        except StopIteration:
                            break
                except ijson.JSONError as exc:
                    raise ValueError(
                        f"nebius JSON must contain an array of valid records: {source}"
                    ) from exc
        return records

    def _task(self, index: int) -> AgentTask:
        try:
            return _normalize_task(self._load()[index])
        except (TypeError, ValueError, KeyError) as exc:
            source = self.path or f"{DATASET_ID}@{self.revision}"
            raise ValueError(
                f"invalid nebius trajectory at {source}, row {index}: {exc}"
            ) from exc


def _text(value: Any, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{field} must be a {'non-empty ' if nonempty else ''}string")
    return value


def _normalize_task(record: dict[str, Any]) -> AgentTask:
    if not isinstance(record, dict):
        raise TypeError("record must be an object")
    trajectory_id = _text(record["trajectory_id"], "trajectory_id", nonempty=True)
    instance_id = _text(record["instance_id"], "instance_id", nonempty=True)
    repo = _text(record["repo"], "repo", nonempty=True)
    exit_status = _text(record["exit_status"], "exit_status", nonempty=True)
    resolved = record["resolved"]
    if resolved not in (0, 1) or not isinstance(resolved, (bool, int)):
        raise ValueError("resolved must be 0 or 1")
    source_messages = record["trajectory"]
    if not isinstance(source_messages, list) or not source_messages:
        raise ValueError("trajectory must be a non-empty list")

    messages: list[AgentMessage] = []
    calls: dict[str, str] = {}
    observations: set[str] = set()
    for position, message in enumerate(source_messages):
        if not isinstance(message, dict):
            raise TypeError(f"message {position} must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {position} has an unsupported role: {role!r}")
        content = _text(message.get("content"), f"message {position} content")
        tool_calls: list[AgentToolCall] = []
        if role == "assistant":
            raw_calls = message.get("tool_calls")
            if raw_calls is not None and not isinstance(raw_calls, list):
                raise ValueError(f"message {position} tool_calls must be a list")
            for call in raw_calls or []:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError(f"message {position} has an invalid tool call")
                call_id = _text(call.get("id"), "tool call id", nonempty=True)
                if call_id in calls:
                    raise ValueError(f"duplicate tool call id: {call_id}")
                function = call.get("function")
                if not isinstance(function, dict):
                    raise TypeError("tool call function must be an object")
                name = _text(function.get("name"), "tool name", nonempty=True)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                else:
                    # Return an independent snapshot even for already decoded input.
                    arguments = json.loads(json.dumps(arguments, allow_nan=False))
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must decode to an object")
                calls[call_id] = name
                tool_calls.append(AgentToolCall(call_id, name, arguments))

        call_id = None
        name = None
        if role == "tool":
            call_id = _text(message.get("tool_call_id"), "tool_call_id", nonempty=True)
            name = _text(message.get("name"), "tool observation name", nonempty=True)
            if calls.get(call_id) != name:
                raise ValueError(f"unmatched tool observation: {call_id} ({name})")
            if call_id in observations:
                raise ValueError(f"duplicate tool observation: {call_id}")
            observations.add(call_id)
        messages.append(AgentMessage(role, content, tuple(tool_calls), call_id, name))

    if not any(message.role == "user" for message in messages):
        raise ValueError("trajectory has no user request")
    if not any(message.role == "assistant" for message in messages):
        raise ValueError("trajectory has no assistant activity")
    # A terminal finish action has no observation. Interrupted attempts can
    # also end with unobserved calls; do not invent the missing results.
    trailing_calls = {call.id for call in messages[-1].tool_calls}
    for call_id, name in calls.items():
        if (
            call_id not in observations
            and name != "finish"
            and (exit_status == "submit" or call_id not in trailing_calls)
        ):
            raise ValueError(f"missing tool observation: {call_id} ({name})")

    final_response = None
    if messages[-1].role == "assistant":
        finish_calls = [c for c in messages[-1].tool_calls if c.name == "finish"]
        if len(finish_calls) > 1:
            raise ValueError("multiple terminal finish calls")
        if finish_calls:
            final_response = _text(
                finish_calls[0].arguments.get("message"), "finish message"
            )
    return AgentTask(
        trajectory_id,
        instance_id,
        repo,
        tuple(messages),
        exit_status,
        bool(resolved),
        final_response,
    )


__all__ = ["NebiusAdapter"]
