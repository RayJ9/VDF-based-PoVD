from __future__ import annotations

import argparse
import json

from .core import (
    build_platform_config,
    compare_mining_modes,
    format_comparison,
    format_mining_result,
    run_mining_platform,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="povd_platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config-file")
    run_parser.add_argument("--mode", choices=["povd", "pow", "vdf_baseline"], required=True)
    run_parser.add_argument("--quiet", action="store_true")
    run_parser.add_argument("--json", action="store_true")
    run_parser.add_argument("--target-height", type=int)
    run_parser.add_argument("--num-miners", type=int)
    run_parser.add_argument("--round-limit", type=int)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--config-file")
    compare_parser.add_argument("--quiet", action="store_true")
    compare_parser.add_argument("--json", action="store_true")
    compare_parser.add_argument("--target-height", type=int)
    compare_parser.add_argument("--num-miners", type=int)
    compare_parser.add_argument("--round-limit", type=int)

    return parser


def build_settings_from_args(args: argparse.Namespace):
    overrides = {}
    if getattr(args, "config_file", None):
        with open(args.config_file, "r", encoding="utf-8") as handle:
            overrides.update(json.load(handle))
    if args.target_height is not None:
        overrides["target_block_height"] = args.target_height
    if args.num_miners is not None:
        overrides["num_miners"] = args.num_miners
    return build_platform_config(**overrides) if overrides else None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = build_settings_from_args(args)

    if args.command == "run":
        result = run_mining_platform(
            mode=args.mode,
            settings=settings,
            verbose=not args.quiet,
            round_limit=args.round_limit,
        )
        print(json.dumps(result, indent=2) if args.json else format_mining_result(result))
        return

    results = compare_mining_modes(settings=settings, verbose=not args.quiet, round_limit=args.round_limit)
    print(json.dumps(results, indent=2) if args.json else format_comparison(results))


if __name__ == "__main__":
    main()
