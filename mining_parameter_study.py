import re
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SETTINGS_PATH = PACKAGE_ROOT / "settings.py"


def update_settings(num_miners, propagation_vector, min_delay, max_delay, pow_probability, target_height):
    content = SETTINGS_PATH.read_text(encoding="utf-8")
    content = re.sub(r"NUM_MINERS = .+", f"NUM_MINERS = {num_miners}", content)
    content = re.sub(r"PROPAGATION_VECTOR_WD = \[.*?\]", f"PROPAGATION_VECTOR_WD = {propagation_vector}", content)
    content = re.sub(r"MAX_PROPAGATION_DELAY_D = .+", "MAX_PROPAGATION_DELAY_D = len(PROPAGATION_VECTOR_WD)", content)
    content = re.sub(
        r"NETWORK_CAPABILITY_C = .+",
        "NETWORK_CAPABILITY_C = sum(PROPAGATION_VECTOR_WD) / MAX_PROPAGATION_DELAY_D if MAX_PROPAGATION_DELAY_D else None",
        content,
    )
    content = re.sub(r"MIN_DELAY_ROUNDS = .+", f"MIN_DELAY_ROUNDS = {min_delay}", content)
    content = re.sub(r"MAX_DELAY_ROUNDS = .+", f"MAX_DELAY_ROUNDS = {max_delay}", content)
    content = re.sub(r"TARGET_BLOCK_HEIGHT = .+", f"TARGET_BLOCK_HEIGHT = {target_height}", content)
    content = re.sub(
        r"POW_MINING_PROBABILITY_PER_HASH = .*",
        f"POW_MINING_PROBABILITY_PER_HASH = {pow_probability:.10f} / POW_HASHES_PER_ROUND",
        content,
    )
    content = re.sub(
        r"POW_DIFFICULTY_PHI = .+",
        "POW_DIFFICULTY_PHI = int(POW_MINING_PROBABILITY_PER_HASH * (2**256))",
        content,
    )
    SETTINGS_PATH.write_text(content, encoding="utf-8")


def run_mining_compare():
    result = subprocess.run(
        [sys.executable, "-m", "povd_platform", "compare"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    povd_rate = 0.0
    pow_rate = 0.0
    for line in result.stdout.split("\n"):
        if line.startswith("Fork Rate"):
            parts = line.split("|")
            if len(parts) >= 3:
                try:
                    povd_rate = float(parts[1].strip())
                    pow_rate = float(parts[2].strip())
                except ValueError:
                    pass
    return povd_rate, pow_rate


def run_mining_parameter_study(num_miners, propagation_vector, target_height, probabilities, sigma):
    if num_miners is None or not propagation_vector or target_height is None or not probabilities or sigma is None:
        raise SystemExit("Fill in mining parameter study inputs before running this module.")

    block_times = [1 / value for value in probabilities]

    print("=== Running Mining Parameter Study ===")
    print(f"Miners (n) = {num_miners}")
    print(f"Max Delay (d) = 10")
    print(f"Network Capability (c) = 0.4")
    print(f"Sigma (Delay Range) = {sigma}")
    print(f"{'Block Time':<12} | {'PoVD Fork Rate':<20} | {'PoW Fork Rate':<20}")
    print("-" * 60)

    for expected_block_time in block_times:
        min_delay = max(1, int(round(expected_block_time - sigma / (num_miners + 1))))
        max_delay = min_delay + sigma
        actual_block_time = min_delay + sigma / (num_miners + 1)
        pow_probability = 1.0 / (num_miners * actual_block_time)
        update_settings(num_miners, propagation_vector, min_delay, max_delay, round(pow_probability, 8), target_height)
        povd_rate, pow_rate = run_mining_compare()
        print(f"{actual_block_time:<12.2f} | {povd_rate:<20.4f} | {pow_rate:<20.4f}")


if __name__ == "__main__":
    num_miners = None
    propagation_vector = []
    target_height = None
    probabilities = []
    sigma = None
    run_mining_parameter_study(num_miners, propagation_vector, target_height, probabilities, sigma)
