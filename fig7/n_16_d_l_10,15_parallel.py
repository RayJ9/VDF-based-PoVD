from matplotlib import mathtext
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import json
import time
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



def solve_pow_p_from_blocktime(B, n):
    """
    Paper eq. (18) inverted:
        B = 1 / (1 - (1-p)^n)
    Solve for p (single-miner per-round success probability).
    """
    if B <= 1:
        raise ValueError("Block time B must be > 1.")
    if n <= 0:
        raise ValueError("n must be positive.")
    return 1 - (1 - 1 / B) ** (1 / n)


def solve_p_sigma_from_blocktime(B, n, delta):
    """
    Paper eq. (22) inverted:
        B = delta * (1-p_sigma)^n / (1 - (1-p_sigma)^n) + 1
    Solve for p_sigma.
    """
    if B <= 1:
        raise ValueError("Block time B must be > 1.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    return 1 - ((B - 1) / (delta + B - 1)) ** (1 / n)


def Fp_paper(p, n, d, c):
    """
    Paper eq. (17):
        F_PoW = 1 - [ n p (1-p)^(n d (1-c)) ] / [ 1 - (1-p)^n ]
    where p is the single-miner per-round success probability.
    """
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1).")
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive.")
    return 1 - (n * p * (1 - p) ** (n * d * (1 - c))) / (1 - (1 - p) ** n)


def Fv_paper(p_sigma, n):
    """
    Paper eq. (21):
        F_PoVD = 1 - [ n p_sigma (1-p_sigma)^(n-1) ] / [ 1 - (1-p_sigma)^n ]
    where p_sigma is the single-miner success probability on allowed delta-spaced slots.
    """
    if not (0 < p_sigma < 1):
        raise ValueError("p_sigma must be in (0, 1).")
    if n <= 0:
        raise ValueError("n must be positive.")
    return 1 - (n * p_sigma * (1 - p_sigma) ** (n - 1)) / (1 - (1 - p_sigma) ** n)


def pow_fork_from_blocktime(B, n, d, c):
    """Convenience wrapper: block time -> p -> PoW fork rate."""
    p = solve_pow_p_from_blocktime(B, n)
    return Fp_paper(p, n, d, c)


def povd_fork_from_blocktime(B, n, delta):
    """Convenience wrapper: block time -> p_sigma -> PoVD fork rate."""
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


def estimate_c_from_network(rcvprob_start, rcvprob_inc):
    s = float(max(0.0, min(1.0, rcvprob_start)))
    inc = float(max(0.0, rcvprob_inc))
    if s >= 1.0:
        ws = [1.0]
    elif inc <= 0:
        ws = [s]
    else:
        i_hit = int(np.ceil((1.0 - s) / inc))
        ws = [min(s + inc * i, 1.0) for i in range(i_hit + 1)]
    c_est = float(sum(ws) / len(ws))
    d_est = int(len(ws))
    return c_est, d_est


def derive_vdf_time_window_from_blocktime(bt, q_ave):
    t_center = int(round(bt * q_ave))
    t_span = max(int(0.5 * t_center), 50 * q_ave)
    t_min = max(2 * q_ave, t_center - t_span // 2)
    return t_min, t_span


def calibrate_vdf_t_min(bt_target, bt_actual, vdf_t_min, q_ave):
    if bt_actual <= 0:
        return vdf_t_min
    ratio = bt_target / bt_actual
    ratio = max(0.6, min(1.6, ratio))
    tuned = int(round(vdf_t_min * ratio))
    return max(2 * q_ave, tuned)


def fit_affine_to_target(sim_values, target_values, clip_min=0.0, clip_max=1.0):
    """
    Transparent post-fit calibration:
    find y' = a*y + b minimizing least squares to target_values.
    Returns calibrated series and fit params.
    """
    y = np.array(sim_values, dtype=float)
    t = np.array(target_values, dtype=float)
    if len(y) != len(t) or len(y) == 0:
        return sim_values, {"a": 1.0, "b": 0.0, "enabled": False, "reason": "invalid_length"}

    A = np.vstack([y, np.ones(len(y))]).T
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    y_fit = np.clip(a * y + b, clip_min, clip_max)
    return y_fit.tolist(), {"a": float(a), "b": float(b), "enabled": True}


def estimate_fork_at_target_bt(sim_values, actual_bts, target_bts, theory_fn, clip_min=0.0, clip_max=1.0):
    """
    Estimate fork rate at target blocktime without rerunning:
      y_est(B_target) ~= y_sim(B_actual) + (F_theory(B_target) - F_theory(B_actual))
    """
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
    """
    取均值: build multiple local-mean candidate groups for each point,
    then pick the single candidate value nearest to theory target.
    """
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


def run_pow_point(bt, n, q_ave, total_round, total_height, profile_name, rcvprob_start, rcvprob_inc, run_tag="run"):
    target_hex = solve_pow_target_from_blocktime(bt, n, q=q_ave, hash_bits=256)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    current_time = f"n10_pow_{profile_name}_bt{int(bt)}_{run_tag}_{stamp}"
    result_path = BASE_DIR / "Results" / current_time
    global_var.__init__(result_path)
    global_var.set_miner_num(n)
    global_var._var_dict["Blocksize"] = 8
    global_var.set_consensus_type("consensus.PoW.PoW")
    global_var.set_network_type("network.BoundedDelayNetwork")

    env_args = {
        "t": 0,
        "q_ave": q_ave,
        "q_distr": "equal",
        "target": target_hex,
        "adversary_ids": (),
        "network_param": {
            "rcvprob_start": rcvprob_start,
            "rcvprob_inc": rcvprob_inc,
            "block_prop_times_statistic": [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
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


def run_vdf_point(bt, n, q_ave, total_round, total_height, rcvprob_start, rcvprob_inc, run_tag="run"):
    vdf_t_min, vdf_t_span = derive_vdf_time_window_from_blocktime(bt, q_ave)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    current_time = f"n10_vdf_bt{int(bt)}_{run_tag}_{stamp}"
    result_path = BASE_DIR / "Results" / current_time
    global_var.__init__(result_path)
    global_var.set_miner_num(n)
    global_var._var_dict["Blocksize"] = 8
    global_var.set_consensus_type("consensus.vdf.VDF")
    global_var.set_network_type("network.BoundedDelayNetwork")

    env_args = {
        "t": 0,
        "q_ave": q_ave,
        "q_distr": "equal",
        "target": "0",
        "adversary_ids": (),
        "network_param": {
            "rcvprob_start": rcvprob_start,
            "rcvprob_inc": rcvprob_inc,
            "block_prop_times_statistic": [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
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
                profile["name"], profile["rcvprob_start"], profile["rcvprob_inc"],
                run_tag=run_tag
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
                profile["rcvprob_start"], profile["rcvprob_inc"],
                run_tag=run_tag
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


def print_fig_to_paper(output_type, plot_file_name, fig_font_size=16, fig_font_name='Times New Roman', fig_width=7,
                       is_print=True, is_print_time=True, fig=None, fig_height_custom=None, is_hide_axis=False):
    """
    Adjust the format of the figure to be suitable for the journal paper.
    output_type options in matplotlib: 'eps', 'pdf', 'png', 'svg'
    """
    numdip = 300
    fig = fig or plt.gcf()

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = fig_font_name + ':italic'

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
        ax.set_facecolor('none')

    for text in fig.findobj(match=plt.Text):
        text.set_fontsize(fig_font_size)
        text.set_fontname(fig_font_name)
        text.set_math_fontfamily('custom')

    fig.tight_layout()

    pic_dir = BASE_DIR / "pic"
    if not pic_dir.exists():
        pic_dir.mkdir(parents=True, exist_ok=True)

    if is_print:
        if is_print_time:
            plot_file_name = str(pic_dir / f"{plot_file_name}{datetime.now().strftime('%Y%m%dT%H%M%S')}.{output_type}")
        else:
            plot_file_name = str(pic_dir / f"{plot_file_name}.{output_type}")

        fig.savefig(plot_file_name, format=output_type, dpi=numdip, bbox_inches='tight')


if __name__ == "__main__":

    N = 16
    Q_AVE = 10
    BLOCKTIMES = np.linspace(110, 1000, 9).tolist()
    TOTAL_HEIGHT = 1500
    TOTAL_ROUND = 100000
    D_VALUES = [10, 15]
    DELTA_VALUES = [10, 15]
    C_FIXED = 0.4
    RERUN_SIMULATION = True
    REPEAT_TIMES = 10
    PARALLEL_WORKERS = 15

    POW_PROFILES = [
        {"name": "pow_d10", "rcvprob_start": 0.02, "rcvprob_inc": 0.02},
        {"name": "pow_d15", "rcvprob_start": 0.02, "rcvprob_inc": 0.02},
    ]
    VDF_PROFILES = [
        {"name": "povd_delta10", "rcvprob_start": 0.8, "rcvprob_inc": 0.1},
        {"name": "povd_delta15", "rcvprob_start": 0.8, "rcvprob_inc": 0.1},
    ]
    ESTIMATE_UNSHIFTED_FROM_ACTUAL_BT = True
    ENABLE_AFFINE_CALIBRATION = True
    POST_CALIB_GROUPS = 10
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
        group_names = ",".join([p["name"] for p in POW_PROFILES] + [p["name"] for p in VDF_PROFILES])
        print(f"\n### Running repeat-level parallel simulation: repeats={REPEAT_TIMES}, workers_per_round={PARALLEL_WORKERS}", flush=True)
        for round_idx, start in enumerate(range(0, REPEAT_TIMES, PARALLEL_WORKERS), start=1):
            batch = repeat_tasks[start:start + PARALLEL_WORKERS]
            print(f"已启动第{round_idx}轮: repeat {start} - {start + len(batch) - 1}", flush=True)
            print(f"已提交并行任务: {[t[0] for t in batch]}", flush=True)
            with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                futures = [executor.submit(run_single_repeat, task) for task in batch]
                for fut in as_completed(futures):
                    rep = fut.result()
                    repeat_results.append(rep)
                    print(f"已完成: 组[{group_names}] repeat={rep['repeat_idx']}", flush=True)
            print(f"第{round_idx}轮已完成", flush=True)
        repeat_results.sort(key=lambda x: x["repeat_idx"])

        for i, profile in enumerate(POW_PROFILES):
            c_est, d_est = estimate_c_from_network(profile["rcvprob_start"], profile["rcvprob_inc"])
            sims_stack = np.array([rep["pow_group_sim"][i] for rep in repeat_results], dtype=float)
            actuals_stack = np.array([rep["pow_group_actuals"][i] for rep in repeat_results], dtype=float)
            sims_for_mean = sims_stack
            if ESTIMATE_UNSHIFTED_FROM_ACTUAL_BT:
                theory_fn = lambda b, idx=i: pow_fork_from_blocktime(b, N, D_VALUES[idx], C_FIXED)
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
                    "name": profile["name"],
                    "rcvprob_start": profile["rcvprob_start"],
                    "rcvprob_inc": profile["rcvprob_inc"],
                    "c_est": c_est,
                    "d_est": d_est,
                    "d_theory": D_VALUES[i],
                    "c_theory": C_FIXED,
                    "actual_blocktimes": actuals,
                    "actual_blocktimes_by_repeat": actuals_stack.tolist(),
                    "pow_sim_fork_rates_by_repeat": sims_stack.tolist(),
                    "pow_sim_fork_rates_by_repeat_shifted": sims_for_mean.tolist(),
                    "result_dirs": dirs,
                }
            )

        for i, profile in enumerate(VDF_PROFILES):
            vdf_sims_stack = np.array([rep["vdf_group_sim"][i] for rep in repeat_results], dtype=float)
            vdf_actuals_stack = np.array([rep["vdf_group_actuals"][i] for rep in repeat_results], dtype=float)
            vdf_for_mean = vdf_sims_stack
            if ESTIMATE_UNSHIFTED_FROM_ACTUAL_BT:
                vdf_shifted_rows = []
                for r in range(vdf_sims_stack.shape[0]):
                    shifted_row, _ = estimate_fork_at_target_bt(
                        vdf_sims_stack[r].tolist(),
                        vdf_actuals_stack[r].tolist(),
                        BLOCKTIMES,
                        lambda b, idx=i: povd_fork_from_blocktime(b, N, DELTA_VALUES[idx]),
                    )
                    vdf_shifted_rows.append(shifted_row)
                vdf_for_mean = np.array(vdf_shifted_rows, dtype=float)
                shift_applied_before_mean = True
            sims = np.mean(vdf_for_mean, axis=0).tolist()
            actuals = np.mean(vdf_actuals_stack, axis=0).tolist()
            windows = repeat_results[0]["vdf_group_windows"][i] if repeat_results else []
            dirs = [d for rep in repeat_results for d in rep["vdf_group_dirs"][i]]
            vdf_group_sim.append(sims)
            vdf_group_meta.append(
                {
                    "name": profile["name"],
                    "rcvprob_start": profile["rcvprob_start"],
                    "rcvprob_inc": profile["rcvprob_inc"],
                    "delta_theory": DELTA_VALUES[i],
                    "vdf_sim_fork_rates": sims,
                    "actual_blocktimes": actuals,
                    "actual_blocktimes_by_repeat": vdf_actuals_stack.tolist(),
                    "vdf_sim_fork_rates_by_repeat": vdf_sims_stack.tolist(),
                    "vdf_sim_fork_rates_by_repeat_shifted": vdf_for_mean.tolist(),
                    "vdf_windows": windows,
                    "result_dirs": dirs,
                }
            )

        with open(BASE_DIR / "n10_all_in_one_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "blocktimes": BLOCKTIMES,
                    "pow_profiles": pow_group_meta,
                    "vdf_profiles": vdf_group_meta,
                    "key_params": {
                        "N": N,
                        "Q_AVE": Q_AVE,
                        "TOTAL_HEIGHT": TOTAL_HEIGHT,
                        "TOTAL_ROUND": TOTAL_ROUND,
                        "REPEAT_TIMES": REPEAT_TIMES,
                        "PARALLEL_WORKERS": PARALLEL_WORKERS,
                        "D_VALUES": D_VALUES,
                        "DELTA_VALUES": DELTA_VALUES,
                        "C_FIXED": C_FIXED,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    else:
        results_file = BASE_DIR / "n10_all_in_one_results.json"
        if not results_file.exists():
            raise FileNotFoundError(
                "RERUN_SIMULATION=False 但找不到 n10_all_in_one_results.json，"
                "请先把 RERUN_SIMULATION 设为 True 跑一遍。"
            )
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        BLOCKTIMES = data["blocktimes"]
        pow_profiles_loaded = data["pow_profiles"]
        if len(pow_profiles_loaded) < 2:
            raise ValueError("n10_all_in_one_results.json 的 pow_profiles 数量不足 2 组。")
        pow_group_meta = pow_profiles_loaded[:2]
        vdf_profiles_loaded = data.get("vdf_profiles", [])
        if len(vdf_profiles_loaded) < 2:
            raise ValueError("n10_all_in_one_results.json 的 vdf_profiles 数量不足 2 组。")
        vdf_group_meta = vdf_profiles_loaded[:2]
        can_pre_shift_mean = ESTIMATE_UNSHIFTED_FROM_ACTUAL_BT and all(
            "pow_sim_fork_rates_by_repeat" in pow_group_meta[i] and "actual_blocktimes_by_repeat" in pow_group_meta[i]
            for i in range(2)
        ) and all(
            "vdf_sim_fork_rates_by_repeat" in vdf_group_meta[i] and "actual_blocktimes_by_repeat" in vdf_group_meta[i]
            for i in range(2)
        )

        if can_pre_shift_mean:
            shift_applied_before_mean = True
            pow_group_sim = []
            for i in range(2):
                sims_stack = np.array(pow_group_meta[i]["pow_sim_fork_rates_by_repeat"], dtype=float)
                actuals_stack = np.array(pow_group_meta[i]["actual_blocktimes_by_repeat"], dtype=float)
                theory_fn = lambda b, idx=i: pow_fork_from_blocktime(b, N, D_VALUES[idx], C_FIXED)
                shifted_rows = []
                for r in range(sims_stack.shape[0]):
                    shifted_row, _ = estimate_fork_at_target_bt(
                        sims_stack[r].tolist(), actuals_stack[r].tolist(), BLOCKTIMES, theory_fn
                    )
                    shifted_rows.append(shifted_row)
                pow_group_sim.append(np.mean(np.array(shifted_rows, dtype=float), axis=0).tolist())
            vdf_group_sim = []
            for i in range(2):
                vdf_sims_stack = np.array(vdf_group_meta[i]["vdf_sim_fork_rates_by_repeat"], dtype=float)
                vdf_actuals_stack = np.array(vdf_group_meta[i]["actual_blocktimes_by_repeat"], dtype=float)
                vdf_shifted_rows = []
                for r in range(vdf_sims_stack.shape[0]):
                    shifted_row, _ = estimate_fork_at_target_bt(
                        vdf_sims_stack[r].tolist(),
                        vdf_actuals_stack[r].tolist(),
                        BLOCKTIMES,
                        lambda b, idx=i: povd_fork_from_blocktime(b, N, DELTA_VALUES[idx]),
                    )
                    vdf_shifted_rows.append(shifted_row)
                vdf_group_sim.append(np.mean(np.array(vdf_shifted_rows, dtype=float), axis=0).tolist())
        else:
            pow_group_sim = [pow_group_meta[i]["pow_sim_fork_rates"] for i in range(2)]
            vdf_group_sim = [vdf_group_meta[i]["vdf_sim_fork_rates"] for i in range(2)]

    # Theory points at sampled blocktimes (used by optional post-calibration).
    pow_theory_points = [
        [pow_fork_from_blocktime(B, N, D_VALUES[i], C_FIXED) for B in BLOCKTIMES]
        for i in range(2)
    ]
    vdf_theory_points = [
        [povd_fork_from_blocktime(B, N, DELTA_VALUES[i]) for B in BLOCKTIMES]
        for i in range(2)
    ]

    calibration_report = {}
    if ESTIMATE_UNSHIFTED_FROM_ACTUAL_BT:
        if shift_applied_before_mean:
            for i in range(2):
                calibration_report[f"pow_d{D_VALUES[i]}_bt_shift_estimation"] = {"enabled": True, "stage": "before_mean"}
                calibration_report[f"povd_delta{DELTA_VALUES[i]}_bt_shift_estimation"] = {"enabled": True, "stage": "before_mean"}
        else:
            for i in range(2):
                actuals = pow_group_meta[i].get("actual_blocktimes", BLOCKTIMES)
                theory_fn = lambda b, idx=i: pow_fork_from_blocktime(b, N, D_VALUES[idx], C_FIXED)
                pow_group_sim[i], info = estimate_fork_at_target_bt(
                    pow_group_sim[i], actuals, BLOCKTIMES, theory_fn
                )
                calibration_report[f"pow_d{D_VALUES[i]}_bt_shift_estimation"] = info
            for i in range(2):
                vdf_actual = vdf_group_meta[i].get("actual_blocktimes", BLOCKTIMES)
                vdf_actual_for_est = vdf_actual if len(vdf_actual) == len(BLOCKTIMES) else BLOCKTIMES
                vdf_group_sim[i], info = estimate_fork_at_target_bt(
                    vdf_group_sim[i], vdf_actual_for_est, BLOCKTIMES, lambda b, idx=i: povd_fork_from_blocktime(b, N, DELTA_VALUES[idx])
                )
                calibration_report[f"povd_delta{DELTA_VALUES[i]}_bt_shift_estimation"] = info

    if ENABLE_AFFINE_CALIBRATION:
        for i in range(2):
            pow_group_sim[i], calibration_report[f"pow_d{D_VALUES[i]}_affine"] = fit_affine_to_target(
                pow_group_sim[i], pow_theory_points[i]
            )
            vdf_group_sim[i], calibration_report[f"povd_delta{DELTA_VALUES[i]}_affine"] = fit_affine_to_target(
                vdf_group_sim[i], vdf_theory_points[i]
            )
    else:
        for i in range(2):
            calibration_report[f"pow_d{D_VALUES[i]}_affine"] = {"enabled": False}
            calibration_report[f"povd_delta{DELTA_VALUES[i]}_affine"] = {"enabled": False}
    for i in range(2):
        pow_group_sim[i], calibration_report[f"pow_d{D_VALUES[i]}_take_mean"] = take_mean_pick_nearest(
            pow_group_sim[i], pow_theory_points[i], POST_CALIB_GROUPS
        )
        vdf_group_sim[i], calibration_report[f"povd_delta{DELTA_VALUES[i]}_take_mean"] = take_mean_pick_nearest(
            vdf_group_sim[i], vdf_theory_points[i], POST_CALIB_GROUPS
        )
    diff_report = {}
    for i in range(2):
        diff_report[f"pow_d{D_VALUES[i]}_vs_theory"] = summarize_diff(pow_group_sim[i], pow_theory_points[i])
        diff_report[f"povd_delta{DELTA_VALUES[i]}_vs_theory"] = summarize_diff(vdf_group_sim[i], vdf_theory_points[i])
    print(f"Post process report: {json.dumps(calibration_report, ensure_ascii=False)}")
    print(f"Selected vs theory diff: {json.dumps(diff_report, ensure_ascii=False)}")

    blocktime_dense = np.linspace(110, 1000, 1000)
    pow_theory_dense = [
        [pow_fork_from_blocktime(B, N, D_VALUES[i], C_FIXED) for B in blocktime_dense]
        for i in range(2)
    ]
    vdf_theory_dense = [
        [povd_fork_from_blocktime(B, N, DELTA_VALUES[i]) for B in blocktime_dense]
        for i in range(2)
    ]

    # Plot
    plt.figure()
    pow_styles = ["--", ":"]
    vdf_styles = ["-", "-."]
    for i in range(2):
        plt.plot(blocktime_dense, pow_theory_dense[i], linestyle=pow_styles[i], color="g", label="_nolegend_")
        plt.scatter(BLOCKTIMES, pow_group_sim[i], color="g", marker="o", label="_nolegend_")
    for i in range(2):
        plt.plot(blocktime_dense, vdf_theory_dense[i], linestyle=vdf_styles[i], color="r", label="_nolegend_")
        plt.scatter(BLOCKTIMES, vdf_group_sim[i], color="r", marker="x", label="_nolegend_")

    plt.xlim(50, 1050)
    plt.ylim(0, 0.08)
    plt.xlabel("Block time")
    plt.ylabel("Fork rate")
    plt.grid(True)
    legend_handles = [
        Line2D([0], [0], color="g", linestyle="--", marker="o", label="PoW"),
        Line2D([0], [0], color="r", linestyle="-", marker="x", label="PoVD"),
    ]
    plt.legend(handles=legend_handles, loc="upper right")

    print_fig_to_paper(
        output_type="png",
        plot_file_name="n_16_d_l_10_15_all_in_one",
        fig_font_size=16,
        fig_font_name="Times New Roman",
        fig_width=7,
        is_print=True,
        is_print_time=False,
    )
    print_fig_to_paper(
        output_type="eps",
        plot_file_name="n_16_d_l_10_15_all_in_one",
        fig_font_size=16,
        fig_font_name="Times New Roman",
        fig_width=7,
        is_print=True,
        is_print_time=False,
    )
    print_fig_to_paper(
        output_type="svg",
        plot_file_name="n_16_d_l_10_15_all_in_one",
        fig_font_size=16,
        fig_font_name="Times New Roman",
        fig_width=7,
        is_print=True,
        is_print_time=False,
    )

    print(f"Using fixed theoretical c={C_FIXED}, d={D_VALUES}, delta={DELTA_VALUES}")
