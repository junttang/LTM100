"""LongMemEval dataset adapter.

LongMemEval (xiaowu0162/longmemeval-cleaned) provides, per sample, a set of
"haystack" conversation sessions (the long context) plus a question/answer.
We map each sample to one virtual user:
  - the sample's haystack turn contents -> that user's memory_stream (add)
  - the sample's question/answer       -> that user's query_stream (search)

When `n_users` exceeds the number of available samples, samples are replicated
onto additional virtual users (different user ids, same underlying content).
The virtual-user count we drive is what matters for load, not the dataset's
own sample count.

Dataset loading mirrors evaluation/retrieval_agent/longmemeval_test.py: prefer
the `datasets` loader, fall back to a direct JSON download via
huggingface_hub when the loader hits schema incompatibilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ltm100.common import DatasetAdapter, MemoryItem, QueryItem, UserId


def _split_chunks(text: str, max_chars: int = 3000) -> list[str]:
    """Split a long string into <=max_chars chunks on word boundaries."""
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_len = len(normalized)
    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            split_at = normalized.rfind(" ", start, end)
            if split_at > start + (max_chars // 2):
                end = split_at
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _collect_turn_contents(sample: dict[str, Any]) -> list[str]:
    """Flatten a sample's haystack_sessions into chunked turn contents."""
    turns: list[str] = []
    for session in sample.get("haystack_sessions", []) or []:
        for turn in session or []:
            content = str(turn.get("content", "")).strip()
            if content:
                turns.extend(_split_chunks(content))
    return turns


class LongMemEvalAdapter:
    """DatasetAdapter over the LongMemEval-cleaned HF dataset."""

    name = "longmemeval"

    def __init__(
        self,
        split: str = "longmemeval_s_cleaned",
        length: int | None = None,
        cache_dir: str | None = None,
        path: str | None = None,
    ) -> None:
        """If `path` is given, load directly from that local JSON file instead
        of downloading from HuggingFace. `split`/`cache_dir` are ignored when
        `path` is set."""
        self.split = split
        self.length = length
        self.cache_dir = cache_dir
        self.path = path
        self._records: list[dict[str, Any]] | None = None

    # -- loading -----------------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records

        if self.path:
            self._records = self._load_local(self.path)
            return self._records

        self._records = self._load_hf()
        return self._records

    def _load_local(self, path: str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise TypeError(f"Expected list data in {path}, got {type(raw).__name__}.")
        records = raw[: self.length] if self.length is not None else raw
        return self._normalize(records)

    def _load_hf(self) -> list[dict[str, Any]]:
        split_file = (
            self.split if self.split.endswith(".json") else f"{self.split}.json"
        )
        records: list[dict[str, Any]] | None = None

        try:
            from datasets import load_dataset

            ds = load_dataset(
                "xiaowu0162/longmemeval-cleaned",
                split=self.split,
                cache_dir=self.cache_dir,
            )
            num_rows = len(ds)
            if self.length is not None:
                num_rows = min(self.length, num_rows)
            records = ds.select(range(num_rows)).to_list()
        except Exception:
            from huggingface_hub import hf_hub_download

            data_path = hf_hub_download(
                repo_id="xiaowu0162/longmemeval-cleaned",
                repo_type="dataset",
                filename=split_file,
                cache_dir=self.cache_dir,
            )
            with open(data_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                raise TypeError(
                    f"Expected list data in {split_file}, got {type(raw).__name__}."
                )
            records = raw[: self.length] if self.length is not None else raw

        self._records = self._normalize(records)
        return self._records

    @staticmethod
    def _normalize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            rec = dict(record)
            rec["question"] = str(rec.get("question", ""))
            rec["answer"] = str(rec.get("answer", ""))
            rec.setdefault("question_type", "unknown")
            rec.setdefault("haystack_sessions", [])
            normalized.append(rec)
        return normalized

    # -- DatasetAdapter ----------------------------------------------------

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        """Map `n_users` virtual users onto samples, replicating if needed."""
        import random

        records = self._load()
        if not records:
            return []

        rng = random.Random(seed)
        # Stable assignment: user i -> sample (i % n_samples), with a shuffled
        # rotation so replicated users don't all map to the same order.
        indices = list(range(len(records)))
        rng.shuffle(indices)

        user_ids: list[UserId] = []
        for i in range(n_users):
            sample_idx = indices[i % len(indices)]
            user_ids.append(f"lme_user_{i:05d}_s{sample_idx:05d}")
        return user_ids

    def _sample_for_user(self, user: UserId) -> dict[str, Any]:
        """Resolve the dataset sample backing a virtual user id."""
        # user id format: lme_user_<uid>_s<sample_idx>
        try:
            sample_idx = int(user.rsplit("_s", 1)[1])
        except (IndexError, ValueError):
            sample_idx = 0
        records = self._load()
        return records[sample_idx % len(records)]

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        sample = self._sample_for_user(user)
        for content in _collect_turn_contents(sample):
            yield MemoryItem(content=content, producer=user)

    def query_stream(self, user: UserId) -> Iterator[QueryItem]:
        sample = self._sample_for_user(user)
        question = str(sample.get("question", "")).strip()
        if not question:
            return
        yield QueryItem(
            query=question,
            top_k=20,
            expected={
                "answer": sample.get("answer", ""),
                "question_type": sample.get("question_type", "unknown"),
                "question_id": sample.get("question_id", ""),
            },
        )


__all__ = ["LongMemEvalAdapter", "_split_chunks", "_collect_turn_contents"]
