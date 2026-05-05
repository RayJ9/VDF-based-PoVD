import json
from pathlib import Path

from povd_platform import MiningPlatformConfig, compare_mining_modes, run_mining_platform


CONFIG_PATH = Path(__file__).with_name("mining_demo_config.json")


def load_config() -> MiningPlatformConfig:
    return MiningPlatformConfig(**json.loads(CONFIG_PATH.read_text(encoding="utf-8")))

def _select_minimal_fields(result: dict[str, object]) -> dict[str, object]:
    return {
        "total_blocks": result["total_blocks"],
        "main_chain_height": result["main_chain_height"],
        "orphan_blocks": result["orphan_blocks"],
        "fork_rate": result["fork_rate"],
    }

def _format_comparison_no_power(results: dict[str, dict[str, object]]) -> str:
    povd_result = results["povd"]
    pow_result = results["pow"]
    baseline_result = results["vdf_baseline"]
    lines = [
        "Metric           | PoVD            | PoW             | VDF_Baseline",
        "-" * 74,
        f"Total Blocks     | {povd_result['total_blocks']:<15} | {pow_result['total_blocks']:<15} | {baseline_result['total_blocks']:<15}",
        f"Main Height      | {povd_result['main_chain_height']:<15} | {pow_result['main_chain_height']:<15} | {baseline_result['main_chain_height']:<15}",
        f"Orphan Blocks    | {povd_result['orphan_blocks']:<15} | {pow_result['orphan_blocks']:<15} | {baseline_result['orphan_blocks']:<15}",
        f"Fork Rate        | {povd_result['fork_rate']:<15.4f} | {pow_result['fork_rate']:<15.4f} | {baseline_result['fork_rate']:<15.4f}",
    ]
    return "\n".join(lines)


def main() -> None:
    settings = load_config()
    povd_result = run_mining_platform("povd", settings=settings, verbose=False)
    comparison = compare_mining_modes(settings=settings, verbose=False)
    print("PoVD example run:")
    print(json.dumps(_select_minimal_fields(povd_result), indent=2))
    print()
    print("Three-mode comparison:")
    print(_format_comparison_no_power(comparison))


if __name__ == "__main__":
    main()
