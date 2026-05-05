import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D


try:
    from plot1 import print_fig_to_paper
except ImportError:
    def print_fig_to_paper(output_type, plot_file_name, **kwargs):
        plt.savefig(f"{plot_file_name}.{output_type}", dpi=300)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="povd_platform.plot.mining_replot_slices")
    parser.add_argument("--data-files", nargs="+", required=True)
    return parser


def plot_mining_slice(data, slice_label, base_name):
    plt.figure(figsize=(10, 6))
    modes = ["PoW", "VDF_Baseline", "PoVD"]
    core_counts = [1, 2, 4, 16]
    alpha_map = {1: 0.3, 2: 0.5, 4: 0.7, 16: 1.0}
    color_map = {"PoW": "red", "VDF_Baseline": "#808000", "PoVD": "blue"}
    sliced_data = {}
    all_values = []

    for mode in modes:
        for core_count in core_counts:
            key = f"{mode}_{core_count}"
            if key not in data:
                key = f"('{mode}', {core_count})"
            if key not in data:
                continue

            measurements = data[key]
            if slice_label == "2min":
                start_index, end_index = 0, len(measurements)
            elif slice_label == "1min":
                start_index, end_index = len(measurements) // 4, 3 * len(measurements) // 4
            else:
                start_index, end_index = 3 * len(measurements) // 8, 5 * len(measurements) // 8

            sliced = measurements[start_index:end_index]
            sliced_data[key] = sliced
            all_values.extend(sliced)

    if not all_values:
        plt.close()
        return

    for mode in modes:
        for core_count in core_counts:
            key = f"{mode}_{core_count}"
            if key not in sliced_data:
                key = f"('{mode}', {core_count})"
            if key not in sliced_data:
                continue
            time_axis = np.linspace(6, 12, len(sliced_data[key]))
            plt.plot(time_axis, sliced_data[key], color=color_map[mode], alpha=alpha_map[core_count], linewidth=1.5)

    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.ylabel("Power (watt)", fontsize=14)
    plt.xlabel("Time (min)", fontsize=14)
    plt.xlim(6, 12)
    plt.ylim(max(30, min(all_values) * 0.9), max(all_values) * 1.05)

    def log_formatter(value, _):
        if value == 40:
            return r"$4 \times 10^1$"
        if value == 60:
            return r"$6 \times 10^1$"
        if value == 100:
            return r"$10^2$"
        if value == 200:
            return r"$2 \times 10^2$"
        if value >= 100 and value % 100 == 0:
            exponent = int(np.log10(value))
            if value in [100, 1000]:
                return f"$10^{{{exponent}}}$"
            return f"${int(value / 10**exponent)} \\times 10^{{{exponent}}}$"
        return f"{int(value)}"

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))
    custom_lines = [Line2D([0], [0], color="blue", lw=2), Line2D([0], [0], color="red", lw=2), Line2D([0], [0], color="#808000", lw=2)]
    plt.legend(custom_lines, ["PoVD", "PoW", "VDF Baseline"], loc="center left", fontsize=10, frameon=False)

    output_folder = "pic/power_slices_new_format"
    os.makedirs(output_folder, exist_ok=True)
    plot_file_name = f"power_slices_new_format/replot_{base_name}_{slice_label}"
    print_fig_to_paper(output_type="png", plot_file_name=plot_file_name, fig=plt.gcf(), is_print_time=False)
    print_fig_to_paper(output_type="eps", plot_file_name=plot_file_name, fig=plt.gcf(), is_print_time=False)
    plt.close()


def replot_mining_power_slices(data_file):
    with open(data_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    base_name = os.path.basename(data_file).replace(".json", "")
    plot_mining_slice(data, "2min", base_name)
    plot_mining_slice(data, "1min", base_name)
    plot_mining_slice(data, "30s", base_name)
    print(f"Saved 3 slices for: {base_name}")


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    for path in sorted(arguments.data_files):
        replot_mining_power_slices(path)
