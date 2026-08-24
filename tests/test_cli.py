"""Tests for config parsing, registry resolution, and the CLI parser."""

from __future__ import annotations

import pytest
import yaml

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.datasets.longmemeval import LongMemEvalAdapter
from ltm100.cli import build_parser
from ltm100.config import build_backend, build_dataset, load_config


def _write_config(tmp_path) -> str:
    cfg = {
        "dataset": {
            "name": "longmemeval",
            "split": "longmemeval_s_cleaned",
            "length": 5,
        },
        "backend": {
            "name": "memmachine",
            "base_url": "http://localhost:8080",
            "org_prefix": "ltm100",
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_load_config_parses_sections(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    assert cfg.dataset.name == "longmemeval"
    assert cfg.dataset.options["split"] == "longmemeval_s_cleaned"
    assert cfg.dataset.options["length"] == 5
    assert cfg.backend.name == "memmachine"
    assert cfg.backend.options["base_url"] == "http://localhost:8080"


def test_build_dataset_resolves_adapter(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    ds = build_dataset(cfg.dataset)
    assert isinstance(ds, LongMemEvalAdapter)
    assert ds.split == "longmemeval_s_cleaned"


def test_build_backend_resolves_adapter(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    backend = build_backend(cfg.backend)
    assert isinstance(backend, MemMachineClient)
    assert backend.org_prefix == "ltm100"


def test_load_config_requires_dataset(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"backend": {"name": "memmachine"}}))
    with pytest.raises(ValueError, match="dataset"):
        load_config(str(p))


def test_load_config_requires_backend(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"dataset": {"name": "longmemeval"}}))
    with pytest.raises(ValueError, match="backend"):
        load_config(str(p))


def test_build_dataset_unknown_raises(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    cfg.dataset.name = "nope"
    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataset(cfg.dataset)


def test_cli_run_requires_termination(tmp_path, monkeypatch):
    # The CLI (not argparse) enforces that `run` needs --duration or --ops.
    from ltm100.cli import main

    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--config",
                _write_config(tmp_path),
                "--scenario",
                "add-load",
            ]
        )


def test_cli_run_parses_args(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            _write_config(tmp_path),
            "--scenario",
            "add-search-mixed",
            "--users",
            "50",
            "--duration",
            "60",
            "--seed",
            "7",
            "--global-concurrency",
            "10",
        ]
    )
    assert args.command == "run"
    assert args.scenario == "add-search-mixed"
    assert args.users == 50
    assert args.duration == 60.0
    assert args.seed == 7
    assert args.global_concurrency == 10


def test_cli_cleanup_subcommand(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        ["cleanup", "--config", _write_config(tmp_path), "--users", "5"]
    )
    assert args.command == "cleanup"
    assert args.users == 5
