from . import format_comparison, format_mining_result, run_mining_platform


def run_mining_mode(mode: str) -> dict[str, object]:
    result = run_mining_platform(mode=mode.lower(), verbose=True)
    print(f"\n--- {result['mode']} Results ---")
    print(format_mining_result(result))
    return result


run = run_mining_mode


if __name__ == "__main__":
    povd_result = run_mining_mode("PoVD")
    pow_result = run_mining_mode("PoW")
    baseline_result = run_mining_mode("VDF_Baseline")
    print("\n=== Final Comparison ===")
    print(format_comparison({"povd": povd_result, "pow": pow_result, "vdf_baseline": baseline_result}))
