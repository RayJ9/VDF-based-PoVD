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
        plt.savefig(f"{plot_file_name}.{output_type}", dpi=300)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="povd_platform.plot.mining_replot")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--duration", required=True, type=int)
    return parser


def replot_mining_power_data(data_file: str, duration: int) -> None:
    with open(data_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    slices = {"2min": (0, 1.0), "1min": (0.375, 0.875), "30s": (0.5, 0.75)}
    modes = ["PoW", "VDF_Baseline", "PoVD"]
    core_counts = [1, 4, 8, 16]
    alpha_map = {1: 0.3, 4: 0.5, 8: 0.7, 16: 1.0}
    color_map = {"PoW": "red", "VDF_Baseline": "#808000", "PoVD": "blue"}

    output_folder = "pic/power_replot_loop3_beautified"
    os.makedirs(output_folder, exist_ok=True)
    plt.rc("font", family="Times New Roman")

    for slice_label, (start_fraction, end_fraction) in slices.items():
        figure, axis = plt.subplots()
        sliced_results = {}
        all_values = []

        for mode in modes:
            for core_count in core_counts:
                key = f"{mode}_{core_count}"
                if key not in data:
                    continue
                measurements = data[key]
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
                key = f"{mode}_{core_count}"
                if key not in sliced_results:
                    continue
                time_axis = np.linspace(6, 12, len(sliced_results[key]))
                plt.plot(time_axis, sliced_results[key], color=color_map[mode], alpha=alpha_map[core_count], linewidth=1.5)

        plt.yscale("log")
        axis.grid(False)
        axis.grid(which="major", color="gray", linestyle="-", linewidth=1, alpha=0.5)
        axis.xaxis.grid(True, which="major", color="gray", linestyle="-", linewidth=1, alpha=0.5)
        axis.set_ylabel("Power (watt)")
        axis.set_xlabel("Time (min)")
        axis.set_xlim(6, 12)
        axis.set_ylim(20, 200)

        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = "Times New Roman"
        plt.rcParams["mathtext.it"] = "Times New Roman:italic"
        plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
        axis.set_yticks([20, 40, 60, 100, 200])
        axis.set_yticklabels(
            [r"$\mathrm{2 \times 10^1}$", r"$\mathrm{4 \times 10^1}$", r"$\mathrm{6 \times 10^1}$", r"$\mathrm{10^2}$", r"$\mathrm{2 \times 10^2}$"],
            fontname="Times New Roman",
        )
        axis.set_yticks([30, 50], minor=True)
        axis.yaxis.grid(True, which="minor", color="gray", linestyle="-", linewidth=1, alpha=0.5)
        axis.set_yticklabels([], minor=True)

        from matplotlib.patches import Ellipse

        arrow_color = "#666666"
        arrow_props_solid = dict(arrowstyle="-|>", linestyle="solid", color=arrow_color, linewidth=0.8, mutation_scale=4)
        arrow_props_dashed = dict(arrowstyle="-|>", linestyle="dashed", color=arrow_color, linewidth=0.8, mutation_scale=4)
        annotation_config = [
            ("1 processor", 6.4, 35, 0.45, 0.5, 6.55, (6.0, 6.4)),
            ("4 processors", 7.8, 35, 0.45, 1.5, 8.0, (8.0, 9.0)),
            ("8 processors", 8.9, 95, 0.35, 0.5, 9.1, (9.0, 10.0)),
            ("16 processors", 10.3, 160, 0.45, -2.0, 10.5, (10.0, 11.0)),
        ]

        for label, text_x, text_y, tail_dx, tail_dy, target_x_mid, (search_min, search_max) in annotation_config:
            core_count = int(label.split()[0])
            key_pow = f"PoW_{core_count}"
            key_vdf = f"VDF_Baseline_{core_count}"
            key_povd = f"PoVD_{core_count}"
            if key_pow not in sliced_results or key_vdf not in sliced_results or key_povd not in sliced_results:
                continue

            arr_pow = sliced_results[key_pow]
            arr_vdf = sliced_results[key_vdf]
            arr_povd = sliced_results[key_povd]
            index_mid = min(int((target_x_mid - 6.0) / 6.0 * len(arr_pow)), len(arr_pow) - 1)
            y_pow = arr_pow[index_mid]
            y_vdf = arr_vdf[index_mid]
            y_mid = (y_pow + y_vdf) / 2.0
            search_start = max(0, int((search_min - 6.0) / 6.0 * len(arr_povd)))
            search_end = min(len(arr_povd), int((search_max - 6.0) / 6.0 * len(arr_povd)))

            if search_end > search_start:
                sub_array = arr_povd[search_start:search_end]
                peak_index = search_start + int(np.argmax(sub_array))
                target_x_povd = 6.0 + (peak_index / len(arr_povd)) * 6.0
                y_povd = arr_povd[peak_index]
            else:
                target_x_povd = target_x_mid
                y_povd = arr_povd[index_mid]

            ellipse_width = 0.12
            ellipse_height = y_mid * 0.35
            tail_x = text_x + tail_dx
            tail_y = text_y - tail_dy

            if core_count == 4:
                plt.annotate("", xy=(target_x_povd, y_povd), xytext=(tail_x, tail_y + 1.5), arrowprops=arrow_props_solid)
                plt.annotate("", xy=(target_x_mid + ellipse_width / 2.0, y_mid), xytext=(tail_x, text_y + 3.5), arrowprops=arrow_props_dashed)
            else:
                plt.annotate("", xy=(target_x_povd, y_povd), xytext=(tail_x, tail_y), arrowprops=arrow_props_solid)
                plt.annotate("", xy=(target_x_mid + ellipse_width / 2.0, y_mid), xytext=(tail_x, tail_y), arrowprops=arrow_props_dashed)

            plt.text(text_x, text_y, label, fontsize=11, ha="left", va="bottom", color=arrow_color)
            axis.add_patch(Ellipse((target_x_mid, y_mid), width=ellipse_width, height=ellipse_height, edgecolor=arrow_color, facecolor="none", linestyle="dashed", linewidth=0.8, zorder=10))

        custom_lines = [Line2D([0], [0], color="blue", lw=2), Line2D([0], [0], color="red", lw=2), Line2D([0], [0], color="#808000", lw=2)]
        plt.legend(custom_lines, ["PoVD", "PoW", "VDF Baseline"], loc="upper left", fontsize=12, frameon=False)
        plot_file_name = f"power_replot_loop3_beautified/power_timeline_loop3_{slice_label}"
        print_fig_to_paper(output_type="png", plot_file_name=plot_file_name, fig=figure, is_print_time=False)
        print_fig_to_paper(output_type="eps", plot_file_name=plot_file_name, fig=figure, is_print_time=False)
        plt.close()

    print("Replots saved to pic/power_replot_loop3_beautified")


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    replot_mining_power_data(arguments.data_file, arguments.duration)
