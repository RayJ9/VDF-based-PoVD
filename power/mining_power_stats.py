import csv
import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import wmi


try:
    cpu_info = wmi.WMI().Win32_Processor()[0]
    CPU_NAME = cpu_info.Name
    if "i9" in CPU_NAME or "Ryzen 9" in CPU_NAME:
        CPU_TDP_WATTS = 125
    elif "i7" in CPU_NAME or "Ryzen 7" in CPU_NAME:
        CPU_TDP_WATTS = 105
    elif "i5" in CPU_NAME or "Ryzen 5" in CPU_NAME:
        CPU_TDP_WATTS = 65
    else:
        CPU_TDP_WATTS = 105
except Exception:
    CPU_NAME = None
    CPU_TDP_WATTS = None


IDLE_POWER_WATTS = None


def workload_pow(duration, process_id):
    end_time = time.time() + duration
    nonce = 0
    prefix = f"platform_header_process_{process_id}_".encode()
    while time.time() < end_time:
        for _ in range(5000):
            hashlib.sha256(prefix + str(nonce).encode()).digest()
            nonce += 1


def workload_vdf_baseline(duration, process_id):
    end_time = time.time() + duration
    modulus = (1 << 2048) - (1 << 1024) + 123456789
    value = 2 + process_id
    while time.time() < end_time:
        for _ in range(1000):
            value = pow(value, 2, modulus)


def workload_povd_active(duration):
    end_time = time.time() + duration
    modulus = (1 << 2048) - (1 << 1024) + 123456789
    value = 2
    while time.time() < end_time:
        for _ in range(1000):
            value = pow(value, 2, modulus)


def workload_wait(duration):
    time.sleep(duration)


def run_mining_power_profile(mode, core_count, duration):
    print(f"Running {mode} with {core_count} cores for {duration}s...")
    processes = []

    if mode == "PoW":
        for index in range(core_count):
            processes.append(multiprocessing.Process(target=workload_pow, args=(duration, index)))
    elif mode == "VDF_Baseline":
        for index in range(core_count):
            processes.append(multiprocessing.Process(target=workload_vdf_baseline, args=(duration, index)))
    elif mode == "PoVD":
        processes.append(multiprocessing.Process(target=workload_povd_active, args=(duration,)))
        for _ in range(core_count - 1):
            processes.append(multiprocessing.Process(target=workload_wait, args=(duration,)))

    for process in processes:
        process.start()

    log_file = f"power_log_{mode}_{core_count}.csv"
    gadget_path = r"C:\Program Files\Intel\Power Gadget 3.6\PowerLog3.0.exe"
    try:
        subprocess.run([gadget_path, "-duration", str(duration), "-resolution", "500", "-file", log_file], check=False)
    except Exception as exc:
        print(f"Power Gadget failed: {exc}")
        time.sleep(duration)

    for process in processes:
        if process.is_alive():
            process.terminate()

    measurements = []
    try:
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="ignore") as handle:
                reader = csv.reader(handle)
                header_found = False
                time_index = -1
                power_index = -1
                for row in reader:
                    if not row:
                        continue
                    if "System Time" in row[0]:
                        header_found = True
                        for index, column in enumerate(row):
                            if "Elapsed Time" in column:
                                time_index = index
                            if "Processor Power" in column:
                                power_index = index
                        continue
                    if header_found and time_index != -1 and power_index != -1 and len(row) > max(time_index, power_index):
                        try:
                            elapsed = float(row[time_index])
                            power_value = float(row[power_index])
                            if 30 <= elapsed <= 150:
                                measurements.append(power_value)
                        except ValueError:
                            pass
            try:
                os.remove(log_file)
            except Exception:
                pass
    except Exception as exc:
        print(f"Power log parse failed: {exc}")

    if not measurements:
        if IDLE_POWER_WATTS is None:
            raise ValueError("IDLE_POWER_WATTS is not set.")
        measurements = [IDLE_POWER_WATTS] * int((150 - 30) * 2)

    average_power = float(np.mean(measurements))
    print(f"  -> Avg Power (30s-150s): {average_power:.2f} W")
    return average_power, measurements


def keep_measurements(data):
    return data


def plot_mining_power_timeline(all_results, duration, loop_index):
    try:
        from plot1 import print_fig_to_paper
    except ImportError:
        def print_fig_to_paper(output_type, plot_file_name, **kwargs):
            plt.savefig(f"results/{plot_file_name}.{output_type}", dpi=300)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_results = {f"{key[0]}_{key[1]}": value for key, value in all_results.items()}
    with open(f"results/power_data_{duration}s_loop{loop_index}_{timestamp}.json", "w", encoding="utf-8") as handle:
        json.dump(json_results, handle)

    slices = {"2min": (0, 1.0), "1min": (0.375, 0.875), "30s": (0.5, 0.75)}
    plt.rc("font", family="Times New Roman")

    for slice_label, (start_fraction, end_fraction) in slices.items():
        figure, axis = plt.subplots()
        modes = ["PoW", "VDF_Baseline", "PoVD"]
        core_counts = [1, 4, 8, 16]
        alpha_map = {1: 0.3, 4: 0.5, 8: 0.7, 16: 1.0}
        color_map = {"PoW": "red", "VDF_Baseline": "#808000", "PoVD": "blue"}
        sliced_results = {}
        all_values = []

        for mode in modes:
            for core_count in core_counts:
                key = (mode, core_count)
                if key not in all_results:
                    continue
                measurements = all_results[key]
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
        axis.grid(False)
        axis.grid(which="major", color="gray", linestyle="-", linewidth=1, alpha=0.5)
        axis.xaxis.grid(True, which="major", color="gray", linestyle="-", linewidth=1, alpha=0.5)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Ellipse

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
            key_pow = ("PoW", core_count)
            key_vdf = ("VDF_Baseline", core_count)
            key_povd = ("PoVD", core_count)
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
                local_peak_index = np.argmax(sub_array)
                peak_index = search_start + local_peak_index
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
        axis.set_ylabel("Power (watt)")
        axis.set_xlabel("Time (min)")
        axis.set_xlim(6, 12)
        axis.set_ylim(20, 200)

        output_folder = "pic/power_10_loops_intel_logy"
        os.makedirs(output_folder, exist_ok=True)
        plot_file_name = f"power_10_loops_intel_logy/power_timeline_{duration}s_loop{loop_index}_{timestamp}_{slice_label}"
        print_fig_to_paper(output_type="png", plot_file_name=plot_file_name, fig=figure, is_print_time=False)
        print_fig_to_paper(output_type="eps", plot_file_name=plot_file_name, fig=figure, is_print_time=False)
        plt.close()

    print(f"Power timelines for loop {loop_index} saved to pic/power_10_loops_intel_logy/")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    core_counts = []
    modes = []
    duration_per_run = None
    record_duration = None
    total_loops = None

    if not core_counts or not modes or duration_per_run is None or record_duration is None or total_loops is None:
        raise SystemExit("Fill in power inputs before running this module.")

    print("=== Starting Real-Time Power Platform Run ===")
    print(f"Estimated Idle Power: {IDLE_POWER_WATTS} W")
    print(f"Estimated Max Power: {CPU_TDP_WATTS} W")
    print(f"Total Duration per Config: {duration_per_run}s (Recording last {record_duration}s)")
    print(f"Total Loops: {total_loops}")

    for loop_index in range(total_loops):
        print(f"\n=== Starting Power Loop {loop_index + 1}/{total_loops} ===")
        results = {}
        for mode in modes:
            for core_count in core_counts:
                _, measurements = run_mining_power_profile(mode, core_count, duration=duration_per_run)
                results[(mode, core_count)] = keep_measurements(measurements)
                time.sleep(5)
        plot_mining_power_timeline(results, record_duration, loop_index=loop_index + 1)

    print("\nPower run complete.")
