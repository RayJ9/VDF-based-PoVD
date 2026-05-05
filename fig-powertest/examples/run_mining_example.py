import json
from pathlib import Path

from povd_platform import MiningPlatformConfig, compare_mining_modes, format_comparison, run_mining_platform


CONFIG_PATH = Path(__file__).with_name("mining_demo_config.json")


def load_config() -> MiningPlatformConfig:
    return MiningPlatformConfig(**json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def main() -> None:
    settings = load_config()
    povd_result = run_mining_platform("povd", settings=settings, verbose=False)
    comparison = compare_mining_modes(settings=settings, verbose=False)
    print("PoVD example run:")
    print(json.dumps(povd_result, indent=2))
    print()
    print("Three-mode comparison:")
    print(format_comparison(comparison))


if __name__ == "__main__":
    main()
