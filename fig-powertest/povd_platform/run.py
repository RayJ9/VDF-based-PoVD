from . import format_mining_result, run_mining_platform


def run_mining_mode(mode: str) -> dict[str, object]:
    result = run_mining_platform(mode=mode, verbose=True)
    print("\n--- Platform Run Ended ---")
    print(format_mining_result(result))
    return result


run = run_mining_mode


if __name__ == "__main__":
    raise SystemExit("Use 'python -m povd_platform run --mode <povd|pow|vdf_baseline>'")
