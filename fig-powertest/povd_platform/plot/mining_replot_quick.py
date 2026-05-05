import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


try:
    from plot1 import print_fig_to_paper
except ImportError:
    def print_fig_to_paper(output_type, plot_file_name, **kwargs):
        output_path = f"{plot_file_name}.{output_type}"
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="povd_platform.plot.mining_replot_quick")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--duration", required=True, type=int)
    parser.add_argument("--loop-index", required=True)
    parser.add_argument("--timestamp", required=True)
    return parser


def replot_mining_power_data_quickly(data_file: str, duration: int, loop_index: str, timestamp: str) -> None:
    print(f"Replotting {data_file}")
    with open(data_file, "r", encoding="utf-8") as handle:
        all_results = json.load(handle)

    parsed_results = {}
    for key, value in all_results.items():
        parts = key.split("_")
        mode = "_".join(parts[:-1])
        core_count = int(parts[-1])
        parsed_results[(mode, core_count)] = value

    slices = {"2min": (0, 1.0), "1min": (0.25, 0.75), "30s": (0.375, 0.625)}

    for slice_label, (start_fraction, end_fraction) in slices.items():
        plt.figure(figsize=(10, 6))
        modes = ["PoW", "VDF_Baseline", "PoVD"]
        core_counts = [1, 4, 8, 16]
        alpha_map = {1: 0.3, 4: 0.5, 8: 0.7, 16: 1.0}
        color_map = {"PoW": "red", "VDF_Baseline": "#808000", "PoVD": "blue"}
        sliced_results = {}
        all_values = []

        for mode in modes:
            for core_count in core_counts:
                key = (mode, core_count)
                if key not in parsed_results:
                    continue
                measurements = parsed_results[key]
                start_index = int(len(measurements) * start_fraction)
                end_index = int(len(measurements) * end_fraction)
                sliced = measurements[start_index:end_index]
                sliced_results[key] = sliced
                all_values.extend(sliced)

        if not all_values:
            plt.close()
            continue

        for mode in modes:
            for core_count in core_counts:
                key = (mode, core_count)
                if key not in sliced_results:
                    continue
                time_axis = np.linspace(6, 12, len(sliced_results[key]))
                plt.plot(time_axis, sliced_results[key], color=color_map[mode], alpha=alpha_map[core_count], linewidth=1.5)

        plt.yscale("log")
        plt.yticks([40, 60, 100, 200], [r"$4 \times 10^1$", r"$6 \times 10^1$", r"$10^2$", r"$2 \times 10^2$"])
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        custom_lines = [Line2D([0], [0], color="blue", lw=2), Line2D([0], [0], color="red", lw=2), Line2D([0], [0], color="#808000", lw=2)]
        plt.legend(custom_lines, ["PoVD", "PoW", "VDF Baseline"], loc="center", bbox_to_anchor=(7, 75), bbox_transform=plt.gca().transData, fontsize=10, frameon=False)
        plt.ylabel("Power (watt)", fontsize=14)
        plt.xlabel("Time (min)", fontsize=14)
        plt.xlim(6, 12)
        plt.ylim(20, 350)

        output_folder = "pic/power_10_loops_intel_logy"
        os.makedirs(output_folder, exist_ok=True)
        plot_file_name = f"{output_folder}/power_timeline_{duration}s_{loop_index}_{timestamp}_{slice_label}_REPLOT"
        print_fig_to_paper(output_type="png", plot_file_name=plot_file_name, fig=plt.gcf(), is_print_time=False)
        print_fig_to_paper(output_type="eps", plot_file_name=plot_file_name, fig=plt.gcf(), is_print_time=False)
        plt.close()

    print("Quick replot complete.")


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    replot_mining_power_data_quickly(arguments.data_file, arguments.duration, arguments.loop_index, arguments.timestamp)
