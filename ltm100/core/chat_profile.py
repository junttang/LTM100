"""User-group workload profiles for the chat-replay scenario."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ltm100.common import UserId

_SETTING_KEYS = {
    "think",
    "search_every",
    "answer_time",
    "user_gap",
    "top_k",
    "concurrent_sessions",
}


@dataclass(frozen=True)
class ChatSettings:
    think: float
    search_every: int
    answer_time: float
    user_gap: float
    top_k: int
    concurrent_sessions: int = 1


@dataclass(frozen=True)
class ChatGroup:
    name: str
    share: float
    settings: ChatSettings


class ChatProfile:
    """Resolved chat defaults, groups, and deterministic user assignment."""

    def __init__(self, groups: list[ChatGroup]) -> None:
        self.groups = groups
        self._assignments: dict[UserId, ChatGroup] = {}

    def assign(self, users: list[UserId], *, seed: int) -> None:
        """Assign every whole-run user before process sharding.

        Seeded shuffling prevents ordered datasets from correlating with a
        group, while largest-remainder allocation makes counts add up exactly.
        """
        shuffled = list(users)
        random.Random(seed).shuffle(shuffled)
        counts = self.counts(len(shuffled))
        assignments: dict[UserId, ChatGroup] = {}
        offset = 0
        for group, count in zip(self.groups, counts):
            for user in shuffled[offset : offset + count]:
                assignments[user] = group
            offset += count
        self._assignments = assignments

    def group_for(self, user: UserId) -> ChatGroup:
        try:
            return self._assignments[user]
        except KeyError as error:
            raise RuntimeError(
                f"chat profile has no group assignment for user {user!r}"
            ) from error

    def counts(self, total: int) -> list[int]:
        exact = [group.share * total for group in self.groups]
        counts = [math.floor(value) for value in exact]
        remainder = total - sum(counts)
        order = sorted(
            range(len(self.groups)),
            key=lambda index: (-(exact[index] - counts[index]), index),
        )
        for index in order[:remainder]:
            counts[index] += 1
        return counts

    def metadata(self, total_users: int) -> dict[str, Any]:
        counts = self.counts(total_users)
        return {
            "version": 1,
            "groups": [
                {
                    "name": group.name,
                    "share": group.share,
                    "users": count,
                    **asdict(group.settings),
                }
                for group, count in zip(self.groups, counts)
            ],
        }


def load_chat_profile(
    path: str | Path,
    *,
    think: float,
    search_every: int,
    answer_time: float,
    user_gap: float,
    top_k: int,
) -> ChatProfile:
    """Load and strictly validate a versioned chat workload profile."""
    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise TypeError("chat profile must be a YAML mapping")

    unknown_root = set(raw) - {"version", "defaults", "groups"}
    if unknown_root:
        raise ValueError(f"chat profile has unknown field(s): {_names(unknown_root)}")
    if raw.get("version") != 1:
        raise ValueError("chat profile 'version' must be 1")

    defaults_raw = raw.get("defaults", {})
    if not isinstance(defaults_raw, dict):
        raise TypeError("chat profile 'defaults' must be a mapping")
    _reject_unknown(defaults_raw, _SETTING_KEYS, "chat profile defaults")
    fallback = {
        "think": think,
        "search_every": search_every,
        "answer_time": answer_time,
        "user_gap": user_gap,
        "top_k": top_k,
        "concurrent_sessions": 1,
    }
    defaults = _settings({**fallback, **defaults_raw}, "chat profile defaults")

    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise ValueError("chat profile 'groups' must be a non-empty list")

    groups: list[ChatGroup] = []
    names: set[str] = set()
    for index, item in enumerate(groups_raw):
        where = f"chat profile groups[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{where} must be a mapping")
        _reject_unknown(item, {"name", "share", *_SETTING_KEYS}, where)
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where}.name must be a non-empty string")
        name = name.strip()
        if name in names:
            raise ValueError(f"chat profile group name {name!r} is duplicated")
        names.add(name)
        share = _number(item.get("share"), f"{where}.share")
        if share <= 0:
            raise ValueError(f"{where}.share must be > 0")
        overrides = {key: value for key, value in item.items() if key in _SETTING_KEYS}
        settings = _settings({**asdict(defaults), **overrides}, where)
        groups.append(ChatGroup(name=name, share=share, settings=settings))

    total_share = sum(group.share for group in groups)
    if not math.isclose(total_share, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"chat profile group shares must sum to 1.0, got {total_share:g}")
    return ChatProfile(groups)


def _settings(raw: dict[str, Any], where: str) -> ChatSettings:
    think = _number(raw["think"], f"{where}.think")
    answer_time = _number(raw["answer_time"], f"{where}.answer_time")
    user_gap = _number(raw["user_gap"], f"{where}.user_gap")
    search_every = _positive_int(raw["search_every"], f"{where}.search_every")
    top_k = _positive_int(raw["top_k"], f"{where}.top_k")
    sessions = _positive_int(
        raw["concurrent_sessions"], f"{where}.concurrent_sessions"
    )
    if think < 0:
        raise ValueError(f"{where}.think must be >= 0")
    if answer_time < 0:
        raise ValueError(f"{where}.answer_time must be >= 0")
    if user_gap < 0:
        raise ValueError(f"{where}.user_gap must be >= 0")
    if sessions != 1:
        raise ValueError(
            f"{where}.concurrent_sessions must be 1 until concurrent chat "
            "session execution is enabled"
        )
    return ChatSettings(
        think=think,
        search_every=search_every,
        answer_time=answer_time,
        user_gap=user_gap,
        top_k=top_k,
        concurrent_sessions=sessions,
    )


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _reject_unknown(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where} has unknown field(s): {_names(unknown)}")


def _names(names: set[str]) -> str:
    return ", ".join(sorted(names))


__all__ = ["ChatGroup", "ChatProfile", "ChatSettings", "load_chat_profile"]
