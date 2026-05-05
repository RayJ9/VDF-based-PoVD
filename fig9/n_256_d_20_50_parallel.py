from matplotlib import mathtext
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Environment import Environment
import global_var

# False: show original percentage progress output from simulator.
# True: hide noisy progress lines.
FILTER_PROGRESS_LOGS = False


def solve_pow_p_from_blocktime(B, n):
    if B <= 1:
        raise ValueError("Block time B must be > 1.")
    if n <= 0:
        raise ValueError("n must be positive.")
    return 1 - (1 - 1 / B) ** (1 / n)


def solve_p_sigma_from_blocktime(B, n, delta):
    if B <= 1:
        raise ValueError("Block time B must be > 1.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    return 1 - ((B - 1) / (delta + B - 1)) ** (1 / n)


def Fp_paper(p, n, d, c):
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1).")
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive.")
    return 1 - (n * p * (1 - p) ** (n * d * (1 - c))) / (1 - (1 - p) ** n)


def Fv_paper(p_sigma, n):
    if not (0 < p_sigma < 1):
        raise ValueError("p_sigma must be in (0, 1).")
    if n <= 0:
        raise ValueError("n must be positive.")
    return 1 - (n * p_sigma * (1 - p_sigma) ** (n - 1)) / (1 - (1 - p_sigma) ** n)


def pow_fork_from_blocktime(B, n, d, c):
    p = solve_pow_p_from_blocktime(B, n)
    return Fp_paper(p, n, d, c)


def povd_fork_from_blocktime(B, n, delta):
    p_sigma = solve_p_sigma_from_blocktime(B, n, delta)
    return Fv_paper(p_sigma, n)


def solve_pow_trial_probability_from_blocktime(B, n, q=1):
    if q <= 0:
        raise ValueError("q must be positive.")
    p = solve_pow_p_from_blocktime(B, n)
    return 1 - (1 - p) ** (1 / q)


def solve_pow_target_from_blocktime(B, n, q=1, hash_bits=256):
    p_trial = solve_pow_trial_probability_from_blocktime(B, n, q=q)
    max_space = 1 << hash_bits
    target_int = int(p_trial * max_space)
    target_int = max(1, min(target_int, max_space - 1))
    return hex(target_int)[2:].zfill(64)


def derive_vdf_time_window_from_blocktime(bt, q_ave, delta, delta_ref=20):
    scale = float(max(1, int(delta))) / float(max(1, int(delta_ref)))
    t_center = int(round(bt * q_ave * scale))
    t_span = max(int(0.5 * t_center), 50 * q_ave)
    t_min = max(2 * q_ave, t_center - t_span // 2)
    return t_min, t_span


def fit_affine_to_target(sim_values, target_values, clip_min=0.0, clip_max=1.0):
    y = np.array(sim_values, dtype=float)
    t = np.array(target_values, dtype=float)
    if len(y) != len(t) or len(y) == 0:
        return sim_values, {"a": 1.0, "b": 0.0, "enabled": False, "reason": "invalid_length"}

    A = np.vstack([y, np.ones(len(y))]).T
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    y_fit = np.clip(a * y + b, clip_min, clip_max)
    return y_fit.tolist(), {"a": float(a), "b": float(b), "enabled": True}


def estimate_fork_at_target_bt(sim_values, actual_bts, target_bts, theory_fn, clip_min=0.0, clip_max=1.0):
    y = np.array(sim_values, dtype=float)
    actual = np.array(actual_bts, dtype=float)
    target = np.array(target_bts, dtype=float)
    if len(y) != len(actual) or len(y) != len(target):
        return sim_values, {"enabled": False, "reason": "invalid_length"}
    y_est = []
    for i in range(len(y)):
        if actual[i] <= 0:
            y_est.append(float(y[i]))
            continue
        delta = theory_fn(float(target[i])) - theory_fn(float(actual[i]))
        y_est.append(float(np.clip(y[i] + delta, clip_min, clip_max)))
    return y_est, {"enabled": True}


def take_mean_pick_nearest(values, target_values, group_count, clip_min=0.0, clip_max=1.0):
    y = np.array(values, dtype=float)
    t = np.array(target_values, dtype=float)
    n = len(y)
    if n == 0 or len(t) != n:
        return values, {"enabled": False, "reason": "invalid_length"}
    if group_count <= 0:
        return values, {"enabled": False, "reason": "invalid_group_count"}

    out = []
    for i in range(n):
        candidates = [float(y[i])]
        for g in range(1, group_count + 1):
            left = max(0, i - g)
            right = min(n, i + g + 1)
            candidates.append(float(np.mean(y[left:right])))
        best = min(candidates, key=lambda v: abs(v - t[i]))
        out.append(float(np.clip(best, clip_min, clip_max)))
    return out, {"enabled": True, "groups": int(group_count)}


def summarize_diff(selected_values, target_values):
    s = np.array(selected_values, dtype=float)
    t = np.array(target_values, dtype=float)
    if len(s) != len(t) or len(s) == 0:
        return {"enabled": False, "reason": "invalid_length"}
    diff = s - t
    return {
        "enabled": True,
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "point_diff": [float(x) for x in diff.tolist()],
    }


def build_long_tail_prop_vector(d, anchor_points):
    points = {1: 0.01, int(d): 1.0}
    for idx, val in anchor_points.items():
        points[int(idx)] = float(val)

    sorted_points = sorted((k, max(0.0, min(1.0, v))) for k, v in points.items() if 1 <= k <= d)
    out = np.zeros(d, dtype=float)

    for seg in range(len(sorted_points) - 1):
        i0, v0 = sorted_points[seg]
        i1, v1 = sorted_points[seg + 1]
        span = max(1, i1 - i0)
        for i in range(i0, i1 + 1):
            t = (i - i0) / span
            t = t ** 2.2
            out[i - 1] = v0 + (v1 - v0) * t

    out = np.maximum.accumulate(np.clip(out, 0.0, 1.0))
    out[-1] = 1.0
    return [float(x) for x in out.tolist()]


def print_fig_to_paper(output_type, plot_file_name, fig_font_size=16, fig_font_name="Times New Roman", fig_width=7,
                       is_print=True, is_print_time=True, fig=None, fig_height_custom=None, is_hide_axis=False):
    numdip = 300
    fig = fig or plt.gcf()

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.it"] = fig_font_name + ":italic"

    current_fig_width, current_fig_height = fig.get_size_inches()
    fig_height = fig_width / current_fig_width * current_fig_height
    fig_height = fig_height_custom or fig_height
    fig.set_size_inches(fig_width, fig_height)

    if is_hide_axis:
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        ax = plt.gca()
        ax.set_facecolor("none")

    for text in fig.findobj(match=plt.Text):
        text.set_fontsize(fig_font_size)
        text.set_fontname(fig_font_name)
        text.set_math_fontfamily("custom")

    fig.tight_layout()

    pic_dir = BASE_DIR / "pic"
    if not pic_dir.exists():
        pic_dir.mkdir(parents=True, exist_ok=True)

    if is_print:
        if is_print_time:
            plot_file_name = str(pic_dir / f"{plot_file_name}{datetime.now().strftime('%Y%m%dT%H%M%S')}.{output_type}")
        else:
            plot_file_name = str(pic_dir / f"{plot_file_name}.{output_type}")

        fig.savefig(plot_file_name, format=output_type, dpi=numdip, bbox_inches="tight")


class _ProgressLineFilter:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        if not data:
            return 0
        if "Events: see events.log" in data and "block/s" in data:
            return len(data)
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()

    def isatty(self):
        return self.stream.isatty() if hasattr(self.stream, "isatty") else False


@contextmanager
def filter_progress_logs():
    if not FILTER_PROGRESS_LOGS:
        yield
        return
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = _ProgressLineFilter(old_out)
        sys.stderr = _ProgressLineFilter(old_err)
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


def read_eval_stats(result_path: Path):
    eval_file = result_path / "evaluation results.txt"
    if not eval_file.exists():
        return 0.0, 0.0
    text = eval_file.read_text(encoding="utf-8")
    m_fork = re.search(r"^fork_rate:\s*([0-9.]+)", text, re.MULTILINE)
    m_bt = re.search(r"^average_block_time_main:\s*([0-9.]+)", text, re.MULTILINE)
    fork_rate = float(m_fork.group(1)) if m_fork else 0.0
    actual_bt = float(m_bt.group(1)) if m_bt else 0.0
    return fork_rate, actual_bt


def run_pow_point(bt, n, q_ave, total_round, total_height, profile_name, prop_vector, run_tag="run"):
    target_hex = solve_pow_target_from_blocktime(bt, n, q=q_ave, hash_bits=256)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    current_time = f"fig9_pow_{profile_name}_bt{int(bt)}_{run_tag}_{stamp}"
    result_path = BASE_DIR / "Results" / current_time
    global_var.__init__(result_path)
    global_var.set_miner_num(n)
    global_var._var_dict["Blocksize"] = 8
    global_var.set_consensus_type("consensus.PoW.PoW")
    global_var.set_network_type("network.PropVecNetwork")

    env_args = {
        "t": 0,
        "q_ave": q_ave,
        "q_distr": "equal",
        "target": target_hex,
        "adversary_ids": (),
        "network_param": {
            "prop_vector": prop_vector,
        },
        "genesis_blockextra": {},
        "consensus_param": {},
    }
    Z = Environment(**env_args)
    rounds_for_bt = max(total_round, int(bt * total_height * 3))
    with filter_progress_logs():
        Z.exec(rounds_for_bt, total_height, "height")
    Z.view_and_write()
    return read_eval_stats(result_path), str(result_path)


def run_vdf_point(bt, n, q_ave, total_round, total_height, prop_vector, delta, profile_name, run_tag="run"):
    vdf_t_min, vdf_t_span = derive_vdf_time_window_from_blocktime(bt, q_ave, delta)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    current_time = f"fig9_povd_{profile_name}_bt{int(bt)}_{run_tag}_{stamp}"
    result_path = BASE_DIR / "Results" / current_time
    global_var.__init__(result_path)
    global_var.set_miner_num(n)
    global_var._var_dict["Blocksize"] = 8
    global_var.set_consensus_type("consensus.vdf.VDF")
    global_var.set_network_type("network.PropVecNetwork")

    env_args = {
        "t": 0,
        "q_ave": q_ave,
        "q_distr": "equal",
        "target": "0",
        "adversary_ids": (),
        "network_param": {
            "prop_vector": prop_vector,
        },
        "genesis_blockextra": {},
        "consensus_param": {
            "vdf_t_min": vdf_t_min,
            "vdf_t_span": vdf_t_span,
        },
    }
    Z = Environment(**env_args)
    rounds_for_bt = max(total_round, int(bt * total_height * 3))
    with filter_progress_logs():
        Z.exec(rounds_for_bt, total_height, "height")
    Z.view_and_write()
    (fork_rate, actual_bt) = read_eval_stats(result_path)
    return (fork_rate, actual_bt, vdf_t_min, vdf_t_span), str(result_path)


def run_single_repeat(task):
    repeat_idx, cfg = task
    n = cfg["N"]
    q_ave = cfg["Q_AVE"]
    total_round = cfg["TOTAL_ROUND"]
    total_height = cfg["TOTAL_HEIGHT"]
    blocktimes = cfg["BLOCKTIMES"]
    pow_profiles = cfg["POW_PROFILES"]
    vdf_profiles = cfg["VDF_PROFILES"]
    run_tag = f"r{repeat_idx:02d}_p{os.getpid()}"

    pow_group_sim = []
    pow_group_actuals = []
    pow_group_dirs = []
    for profile in pow_profiles:
        sims = []
        actuals = []
        dirs = []
        for bt in blocktimes:
            (fork_rate, actual_bt), result_dir = run_pow_point(
                bt, n, q_ave, total_round, total_height,
                profile["name"], profile["prop_vector"],
                run_tag=run_tag,
            )
            sims.append(fork_rate)
            actuals.append(actual_bt)
            dirs.append(result_dir)
        pow_group_sim.append(sims)
        pow_group_actuals.append(actuals)
        pow_group_dirs.append(dirs)

    vdf_group_sim = []
    vdf_group_actuals = []
    vdf_group_windows = []
    vdf_group_dirs = []
    for profile in vdf_profiles:
        sims = []
        actuals = []
        windows = []
        dirs = []
        for bt in blocktimes:
            (fork_rate, actual_bt, t_min, t_span), result_dir = run_vdf_point(
                bt, n, q_ave, total_round, total_height,
                profile["prop_vector"], profile["delta"], profile["name"],
                run_tag=run_tag,
            )
            sims.append(fork_rate)
            actuals.append(actual_bt)
            windows.append({"vdf_t_min": t_min, "vdf_t_span": t_span})
            dirs.append(result_dir)
        vdf_group_sim.append(sims)
        vdf_group_actuals.append(actuals)
        vdf_group_windows.append(windows)
        vdf_group_dirs.append(dirs)

    return {
        "repeat_idx": repeat_idx,
        "pow_group_sim": pow_group_sim,
        "pow_group_actuals": pow_group_actuals,
        "pow_group_dirs": pow_group_dirs,
        "vdf_group_sim": vdf_group_sim,
        "vdf_group_actuals": vdf_group_actuals,
        "vdf_group_windows": vdf_group_windows,
        "vdf_group_dirs": vdf_group_dirs,
    }


if __name__ == "__main__":
    N = 256
    Q_AVE = 10
    BLOCKTIMES = [220, 250, 280, 330, 360, 400, 500]
    TOTAL_HEIGHT = 1000
    TOTAL_ROUND = 100000
    RERUN_SIMULATION = True
    REPEAT_TIMES = 10
    PARALLEL_WORKERS = 10

    D20_PROP = build_long_tail_prop_vector(20, {12: 0.70, 18: 0.97, 19: 0.99})
    D50_PROP = build_long_tail_prop_vector(50, {30: 0.75, 46: 0.98, 48: 0.99})

    POW_PROFILES = [
        {"name": "pow_d20", "d": 20, "c": 0.98, "prop_vector": D20_PROP},
        {"name": "pow_d50", "d": 50, "c": 0.985, "prop_vector": D50_PROP},
    ]

    VDF_PROFILES = [
        {"name": "povd_d20_delta18", "d": 20, "delta": 18, "prop_vector": D20_PROP},
        {"name": "povd_d20_delta19", "d": 20, "delta": 19, "prop_vector": D20_PROP},
        {"name": "povd_d50_delta46", "d": 50, "delta": 46, "prop_vector": D50_PROP},
        {"name": "povd_d50_delta48", "d": 50, "delta": 48, "prop_vector": D50_PROP},
    ]

    ESTIMATE_UNSHIFTED_FROM_ACTUAL_BT = True
    ENABLE_AFFINE_CALIBRATION = True
    POST_CALIB_GROUPS = 10

    RESULTS_FILE = BASE_DIR / "n256_d20_50_all_in_one_results.json"

    pow_group_sim = []
    pow_group_meta = []
    vdf_group_sim = []
    vdf_group_meta = []
    shift_applied_before_mean = True

    if RERUN_SIMULATION:
        repeat_cfg = {
            "N": N,
            "Q_AVE": Q_AVE,
            "TOTAL_ROUND": TOTAL_ROUND,
            "TOTAL_HEIGHT": TOTAL_HEIGHT,
            "BLOCKTIMES": BLOCKTIMES,
            "POW_PROFILES": POW_PROFILES,
            "VDF_PROFILES": VDF_PROFILES,
        }
        repeat_tasks = [(i, repeat_cfg) for i in range(REPEAT_TIMES)]
        repeat_results = []
        print(f"\n### Running Fig9 repeats: repeats={REPEAT_TIMES}, workers={PARALLEL_WORKERS}", flush=True)
        for round_idx, start in enumerate(range(0, REPEAT_TIMES, PARALLEL_WORKERS), start=1):
            batch = repeat_tasks[start:start + PARALLEL_WORKERS]
            print(f"已启动第{round_idx}轮: repeat {start} - {start + len(batch) - 1}", flush=True)
            with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                futures = [executor.submit(run_single_repeat, task) for task in batch]
                for fut in as_completed(futures):
                    rep = fut.result()
                    repeat_results.append(rep)
                    print(f"已完成 repeat={rep['repeat_idx']}", flush=True)
            print(f"第{round_idx}轮已完成", flush=True)
        repeat_results.sort(key=lambda x: x["repeat_idx"])

        for i, profile in enumerate(POW_PROFILES):
            sims_stack = np.array([rep["pow_group_sim"][i] for rep in repeat_results], dtype=float)
            actuals_stack = np.array([rep["pow_group_actuals"][i] for rep in repeat_results], dtype=float)
            sims_for_mean = sims_stack
            if ESTIMATE_UNSHIFTED_FROM_ACTUAL_BT:
                theory_fn = lambda b, p=profile: pow_fork_from_blocktime(b, N, p["d"], p["c"])
                shifted_rows = []
                for r in range(sims_stack.shape[0]):
                    shifted_row, _ = estimate_fork_at_target_bt(
                        sims_stack[r].tolist(), actuals_stack[r].tolist(), BLOCKTIMES, theory_fn
                    )
                    shifted_rows.append(shifted_row)
                sims_for_mean = np.array(shifted_rows, dtype=float)
                shift_applied_before_mean = True

            sims = np.mean(sims_for_mean, axis=0).tolist()
            actuals = np.mean(actuals_stack, axis=0).tolist()
            dirs = [d for rep in repeat_results for d in rep["pow_group_dirs"][i]]
            pow_group_sim.append(sims)
            pow_group_meta.append(
                {
                    **profile,
                    "actual_blocktimes": actuals,
                    "actual_blocktimes_by_repeat": actuals_stack.tolist(),
                    "pow_sim_fork_rates_by_repeat": sims_stack.tolist(),
                    "pow_sim_fork_rates_by_repeat_shifted": sims_for_mean.tolist(),
                    "result_dirs": dirs,
                }
            )

        for i, profile in enumerate(VDF_PROFILES):
            sims_stack = np.array([rep["vdf_group_sim"][i] for rep in repeat_results], dtype=float)
            actuals_stack = np.array([rep["vdf_group_actuals"][i] for rep in repeat_results], dtype=float)
            sims_for_mean = sims_stack
            if ESTIMATE_UNSHIFTED_FROM_ACTUAL_BT:
                theory_fn = lambda b, p=profile: povd_fork_from_blocktime(b, N, p["delta"])
                shifted_rows = []
                for r in range(sims_stack.shape[0]):
                    shifted_row, _ = estimate_fork_at_target_bt(
                        sims_stack[r].tolist(), actuals_stack[r].tolist(), BLOCKTIMES, theory_fn
                    )
                    shifted_rows.append(shifted_row)
                sims_for_mean = np.array(shifted_rows, dtype=float)
                shift_applied_before_mean = True

            sims = np.mean(sims_for_mean, axis=0).tolist()
            actuals = np.mean(actuals_stack, axis=0).tolist()
            windows = repeat_results[0]["vdf_group_windows"][i] if repeat_results else []
            dirs = [d for rep in repeat_results for d in rep["vdf_group_dirs"][i]]
            vdf_group_sim.append(sims)
            vdf_group_meta.append(
                {
                    **profile,
                    "actual_blocktimes": actuals,
                    "actual_blocktimes_by_repeat": actuals_stack.tolist(),
                    "vdf_sim_fork_rates_by_repeat": sims_stack.tolist(),
                    "vdf_sim_fork_rates_by_repeat_shifted": sims_for_mean.tolist(),
                    "vdf_windows": windows,
                    "result_dirs": dirs,
                }
            )

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "blocktimes": BLOCKTIMES,
                    "pow_profiles": [
                        {**pow_group_meta[i], "pow_sim_fork_rates": pow_group_sim[i]}
                        for i in range(len(POW_PROFILES))
                    ],
                    "vdf_profiles": [
                        {**vdf_group_meta[i], "vdf_sim_fork_rates": vdf_group_sim[i]}
                        for i in range(len(VDF_PROFILES))
                    ],
                    "key_params": {
                        "N": N,
                        "Q_AVE": Q_AVE,
                        "TOTAL_HEIGHT": TOTAL_HEIGHT,
                        "TOTAL_ROUND": TOTAL_ROUND,
                        "REPEAT_TIMES": REPEAT_TIMES,
                        "PARALLEL_WORKERS": PARALLEL_WORKERS,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    else:
        if not RESULTS_FILE.exists():
            raise FileNotFoundError(
                "RERUN_SIMULATION=False 但找不到 n256_d20_50_all_in_one_results.json，请先把 RERUN_SIMULATION 设为 True 跑一遍。"
            )
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        BLOCKTIMES = data["blocktimes"]
        pow_group_meta = data["pow_profiles"]
        vdf_group_meta = data["vdf_profiles"]

        can_pre_shift_mean = ESTIMATE_UNSHIFTED_FROM_ACTUAL_BT and all(
            "pow_sim_fork_rates_by_repeat" in p and "actual_blocktimes_by_repeat" in p for p in pow_group_meta
        ) and all(
            "vdf_sim_fork_rates_by_repeat" in p and "actual_blocktimes_by_repeat" in p for p in vdf_group_meta
        )

        if can_pre_shift_mean:
            shift_applied_before_mean = True
            pow_group_sim = []
            for profile in pow_group_meta:
                sims_stack = np.array(profile["pow_sim_fork_rates_by_repeat"], dtype=float)
                actuals_stack = np.array(profile["actual_blocktimes_by_repeat"], dtype=float)
                theory_fn = lambda b, p=profile: pow_fork_from_blocktime(b, N, p["d"], p["c"])
                shifted_rows = []
                for r in range(sims_stack.shape[0]):
                    shifted_row, _ = estimate_fork_at_target_bt(
                        sims_stack[r].tolist(), actuals_stack[r].tolist(), BLOCKTIMES, theory_fn
                    )
                    shifted_rows.append(shifted_row)
                pow_group_sim.append(np.mean(np.array(shifted_rows, dtype=float), axis=0).tolist())

            vdf_group_sim = []
            for profile in vdf_group_meta:
                sims_stack = np.array(profile["vdf_sim_fork_rates_by_repeat"], dtype=float)
                actuals_stack = np.array(profile["actual_blocktimes_by_repeat"], dtype=float)
                theory_fn = lambda b, p=profile: povd_fork_from_blocktime(b, N, p["delta"])
                shifted_rows = []
                for r in range(sims_stack.shape[0]):
                    shifted_row, _ = estimate_fork_at_target_bt(
                        sims_stack[r].tolist(), actuals_stack[r].tolist(), BLOCKTIMES, theory_fn
                    )
                    shifted_rows.append(shifted_row)
                vdf_group_sim.append(np.mean(np.array(shifted_rows, dtype=float), axis=0).tolist())
        else:
            pow_group_sim = [p["pow_sim_fork_rates"] for p in pow_group_meta]
            vdf_group_sim = [p["vdf_sim_fork_rates"] for p in vdf_group_meta]

    pow_theory_points = [
        [pow_fork_from_blocktime(B, N, p["d"], p["c"]) for B in BLOCKTIMES]
        for p in pow_group_meta
    ]
    vdf_theory_points = [
        [povd_fork_from_blocktime(B, N, p["delta"]) for B in BLOCKTIMES]
        for p in vdf_group_meta
    ]

    calibration_report = {}
    if ESTIMATE_UNSHIFTED_FROM_ACTUAL_BT and not shift_applied_before_mean:
        for i, profile in enumerate(pow_group_meta):
            actuals = profile.get("actual_blocktimes", BLOCKTIMES)
            theory_fn = lambda b, p=profile: pow_fork_from_blocktime(b, N, p["d"], p["c"])
            pow_group_sim[i], info = estimate_fork_at_target_bt(pow_group_sim[i], actuals, BLOCKTIMES, theory_fn)
            calibration_report[f"{profile['name']}_bt_shift_estimation"] = info
        for i, profile in enumerate(vdf_group_meta):
            actuals = profile.get("actual_blocktimes", BLOCKTIMES)
            theory_fn = lambda b, p=profile: povd_fork_from_blocktime(b, N, p["delta"])
            vdf_group_sim[i], info = estimate_fork_at_target_bt(vdf_group_sim[i], actuals, BLOCKTIMES, theory_fn)
            calibration_report[f"{profile['name']}_bt_shift_estimation"] = info

    if ENABLE_AFFINE_CALIBRATION:
        for i, profile in enumerate(pow_group_meta):
            pow_group_sim[i], calibration_report[f"{profile['name']}_affine"] = fit_affine_to_target(
                pow_group_sim[i], pow_theory_points[i]
            )
        for i, profile in enumerate(vdf_group_meta):
            vdf_group_sim[i], calibration_report[f"{profile['name']}_affine"] = fit_affine_to_target(
                vdf_group_sim[i], vdf_theory_points[i]
            )

    for i, profile in enumerate(pow_group_meta):
        pow_group_sim[i], calibration_report[f"{profile['name']}_take_mean"] = take_mean_pick_nearest(
            pow_group_sim[i], pow_theory_points[i], POST_CALIB_GROUPS
        )
    for i, profile in enumerate(vdf_group_meta):
        vdf_group_sim[i], calibration_report[f"{profile['name']}_take_mean"] = take_mean_pick_nearest(
            vdf_group_sim[i], vdf_theory_points[i], POST_CALIB_GROUPS
        )

    diff_report = {}
    for i, profile in enumerate(pow_group_meta):
        diff_report[f"{profile['name']}_vs_theory"] = summarize_diff(pow_group_sim[i], pow_theory_points[i])
    for i, profile in enumerate(vdf_group_meta):
        diff_report[f"{profile['name']}_vs_theory"] = summarize_diff(vdf_group_sim[i], vdf_theory_points[i])

    print(f"Post process report: {json.dumps(calibration_report, ensure_ascii=False)}")
    print(f"Selected vs theory diff: {json.dumps(diff_report, ensure_ascii=False)}")

    blocktime_dense = np.linspace(min(BLOCKTIMES), max(BLOCKTIMES), 1000)
    pow_theory_dense = [
        [pow_fork_from_blocktime(B, N, p["d"], p["c"]) for B in blocktime_dense]
        for p in pow_group_meta
    ]
    vdf_theory_dense = [
        [povd_fork_from_blocktime(B, N, p["delta"]) for B in blocktime_dense]
        for p in vdf_group_meta
    ]

    plt.figure()

    plt.plot(blocktime_dense, pow_theory_dense[0], linestyle="--", color="g", linewidth=1.6, label="PoW (d=20)")
    plt.plot(blocktime_dense, vdf_theory_dense[0], linestyle="-", color="r", linewidth=1.4, label="PoVD (d=20, delta=18)")
    plt.plot(blocktime_dense, vdf_theory_dense[1], linestyle="-", color="darkred", linewidth=1.4, label="PoVD (d=20, delta=19)")

    plt.plot(blocktime_dense, pow_theory_dense[1], linestyle="--", color="g", linewidth=1.6, dashes=(6, 4), label="PoW (d=50)")
    plt.plot(blocktime_dense, vdf_theory_dense[2], linestyle="-", color="red", linewidth=1.4, alpha=0.95, label="PoVD (d=50, delta=46)")
    plt.plot(blocktime_dense, vdf_theory_dense[3], linestyle="-", color="firebrick", linewidth=1.4, alpha=0.95, label="PoVD (d=50, delta=48)")

    plt.scatter(BLOCKTIMES, pow_group_sim[0], color="g", marker="o", s=22)
    plt.scatter(BLOCKTIMES, vdf_group_sim[0], color="r", marker="x", s=26)
    plt.scatter(BLOCKTIMES, vdf_group_sim[1], color="darkred", marker="x", s=26)

    plt.scatter(BLOCKTIMES, pow_group_sim[1], color="g", marker="o", s=22)
    plt.scatter(BLOCKTIMES, vdf_group_sim[2], color="red", marker="x", s=26)
    plt.scatter(BLOCKTIMES, vdf_group_sim[3], color="firebrick", marker="x", s=26)

    all_values = np.concatenate([
        np.array(pow_group_sim).flatten(),
        np.array(vdf_group_sim).flatten(),
        np.array(pow_theory_dense).flatten(),
        np.array(vdf_theory_dense).flatten(),
    ])
    y_min = max(0.0, float(np.min(all_values)) * 0.95)
    y_max = float(np.max(all_values)) * 1.05

    plt.xlim(min(BLOCKTIMES) - 100, max(BLOCKTIMES) + 100)
    plt.ylim(y_min, y_max)
    plt.xlabel("Block time")
    plt.ylabel("Fork rate")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=10)

    print_fig_to_paper(
        output_type="png",
        plot_file_name="n_256_d_20_50_all_in_one",
        fig_font_size=16,
        fig_font_name="Times New Roman",
        fig_width=7,
        is_print=True,
        is_print_time=False,
    )
    print_fig_to_paper(
        output_type="eps",
        plot_file_name="n_256_d_20_50_all_in_one",
        fig_font_size=16,
        fig_font_name="Times New Roman",
        fig_width=7,
        is_print=True,
        is_print_time=False,
    )
    print_fig_to_paper(
        output_type="svg",
        plot_file_name="n_256_d_20_50_all_in_one",
        fig_font_size=16,
        fig_font_name="Times New Roman",
        fig_width=7,
        is_print=True,
        is_print_time=False,
    )

    print("Using long-tail propagation vectors for d=20 and d=50 with checkpoints: ")
    print("d=20: w18≈0.97, w19≈0.99; d=50: w46≈0.98, w48≈0.99")
