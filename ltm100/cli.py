"""LTM100 command-line interface.

Top-level commands:
  run      Drive a benchmark run: provision users, run a scenario, report.
  cleanup  Delete per-user state for a run (without running).

Per-run parameters come from CLI flags; adapter choices and endpoint/auth
come from the YAML config file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from ltm100.config import build_backend, build_dataset, load_config
from ltm100.core.config import RunConfig
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import get_scenario
from ltm100.metrics.report import write_raw_ndjson, write_summary_csv, write_summary_json


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        users=args.users,
        seed=args.seed,
        duration=args.duration,
        ops=args.ops,
        global_concurrency=args.global_concurrency,
        warmup=args.warmup,
        rampup=args.rampup,
        preingest=args.preingest,
        preingest_fraction=args.preingest_fraction,
        model=args.model,
        arrival_rate=args.arrival_rate,
        session_ops=args.session_ops,
        queue_bound=args.queue_bound,
        delete_on_exit=not args.no_delete_on_exit,
    )


def _build_scenario(args: argparse.Namespace):
    kwargs: dict[str, Any] = {}
    if args.scenario == "realistic":
        kwargs["search_weight"] = args.search_weight
        kwargs["think"] = args.think
    elif args.scenario == "add-search-mixed":
        kwargs["search_every"] = args.search_every
        kwargs["add_batch"] = args.add_batch
    elif args.scenario == "chat-replay":
        kwargs["think"] = args.think
    return get_scenario(args.scenario, **kwargs)


async def _run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dataset = build_dataset(cfg.dataset)
    backend = build_backend(cfg.backend)
    run_cfg = _build_run_config(args)
    scenario = _build_scenario(args)

    runner = LoadRunner(
        client=backend, dataset=dataset, scenario=scenario, config=run_cfg
    )

    # Open the backend's transport once for the whole run (setup + measured
    # run + teardown). The runner calls setup itself.
    async with backend:  # type: ignore[arg-type]
        await runner.run()
        summary = runner.recorder.summary()
        if run_cfg.delete_on_exit:
            await backend.teardown(
                dataset.users(run_cfg.users, seed=run_cfg.seed), delete=True
            )

    meta = {
        "dataset": cfg.dataset.name,
        "backend": cfg.backend.name,
        "scenario": args.scenario,
        "users": run_cfg.users,
        "seed": run_cfg.seed,
        "duration": run_cfg.duration,
        "ops": run_cfg.ops,
        "global_concurrency": run_cfg.global_concurrency,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    print(json.dumps({"meta": meta, "summary": summary}, indent=2))

    if args.output:
        out = args.output.rstrip("/")
        write_summary_json(summary, f"{out}/summary.json", meta=meta)
        write_summary_csv(summary, f"{out}/summary.csv")
        if args.raw:
            write_raw_ndjson(runner.recorder.raw(), f"{out}/raw.ndjson")
        print(f"reports written to {out}/")

    return 0


async def _cleanup(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dataset = build_dataset(cfg.dataset)
    backend = build_backend(cfg.backend)
    users = dataset.users(args.users, seed=args.seed)
    async with backend:  # type: ignore[arg-type]
        await backend.teardown(users, delete=True)
    print(f"deleted state for {len(users)} users")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ltm100", description="LTM100 load benchmark.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="path to YAML config")
    common.add_argument("--users", type=int, default=10, help="virtual users")
    common.add_argument("--seed", type=int, default=0, help="RNG seed")

    run = sub.add_parser("run", parents=[common], help="run a benchmark")
    run.add_argument("--scenario", required=True, help="scenario name")
    g = run.add_mutually_exclusive_group()
    g.add_argument("--duration", type=float, default=0.0, help="run seconds (0=off)")
    g.add_argument("--ops", type=int, default=0, help="total ops cap (0=off)")
    run.add_argument("--global-concurrency", type=int, default=0, help="max in-flight")
    run.add_argument("--warmup", type=float, default=0.0, help="warmup seconds")
    run.add_argument("--preingest", action="store_true", help="pre-ingest memories before run")
    run.add_argument(
        "--preingest-fraction",
        type=float,
        default=1.0,
        help="fraction of each user's memories to pre-ingest",
    )
    run.add_argument("--rampup", type=float, default=0.0, help="ramp-up seconds")
    run.add_argument(
        "--model",
        choices=("closed", "open"),
        default="closed",
        help="load model (closed=fixed N looping users; open=Poisson arrivals)",
    )
    run.add_argument(
        "--arrival-rate",
        type=float,
        default=0.0,
        help="open model: user arrivals per second (Poisson lambda)",
    )
    run.add_argument(
        "--session-ops",
        type=int,
        default=0,
        help="open model: ops each arriving user performs before leaving",
    )
    run.add_argument(
        "--queue-bound",
        type=int,
        default=0,
        help="open model: max queued beyond cap before rejection (0=reject on cap)",
    )
    run.add_argument(
        "--search-weight",
        type=float,
        default=0.8,
        help="realistic scenario: fraction of ops that are search (0..1)",
    )
    run.add_argument(
        "--think",
        type=float,
        default=0.05,
        help="realistic/chat-replay: max think-time jitter per op (seconds)",
    )
    run.add_argument(
        "--search-every",
        type=int,
        default=20,
        help="add-search-mixed: issue one search after this many adds",
    )
    run.add_argument(
        "--add-batch",
        type=int,
        default=1,
        help="add-search-mixed: memory items batched per add op",
    )
    run.add_argument("--output", default=None, help="output dir for reports")
    run.add_argument("--raw", action="store_true", help="also write raw.ndjson")
    run.add_argument("--no-delete-on-exit", action="store_true", help="keep user state")
    run.set_defaults(func=_run)

    clean = sub.add_parser("cleanup", parents=[common], help="delete per-user state")
    clean.set_defaults(func=_cleanup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not getattr(args, "duration", 0) and not getattr(args, "ops", 0):
        if args.command == "run":
            parser.error("run requires either --duration or --ops")
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
