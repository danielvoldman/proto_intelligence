# proto_intelligence.py
"""
SPEC LOCK — Frozen Execution Blueprint (do not modify behavior unless spec changes)

This script implements EXACTLY the frozen specification:
- 32x32 toroidal grid, 360 steps with phases: warmup(0..99), perturb(100..109), observe(110..359)
- Neighborhood mean over 9 cells (self + 8 Moore neighbors) using wrap-around, no in-place updates
- Update order per timestep:
  1) fast_next from fast_prev and slow_prev
  2) apply perturbation to fast_next during perturb phase (fast only, no extra clipping)
  3) slow_next from slow_prev and fast_next
- Conditions:
  * full (tanh, beta=0.10)
  * no_feedback (tanh, beta=0.0)
  * no_nonlinearity (identity, beta=0.10, still clips fast)
  * optional frozen_slow exists only if ENABLE_FROZEN_SLOW_CONDITION=True (default False); not scheduled by default
- Metrics (frozen):
  * History dependence separation on fast only at t_end+tau for tau in {10,50,150}
  * global_stat(t) = mean(abs(fast(t)))
  * pred_error(t) = mean(abs(slow_prev - fast_next)) computed immediately after fast_next (pre-perturb, pre-slow)
- Output:
  * results/master_metrics_<RUN_TAG>.csv (fixed columns/order)
  * results/timeseries_<RUN_TAG>.csv (fixed columns/order; per timestep for every run)
  * figures/fig1..fig5 PNGs from CSV only; fig6 optional (default off)
  * runlog/run_<RUN_TAG>.json with frozen parameters and derived seeds
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import sys
import zlib
from typing import Dict, List, Tuple, Any

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe; saves PNGs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -----------------------------
# Frozen constants (spec)
# -----------------------------

GRID = 32
BOUNDARY = "wrap"

STEPS = 360
PHASE_WARMUP = 100
PHASE_PERTURB = 10
PHASE_OBSERVE = 250

FAST_INIT_LOW = -0.05
FAST_INIT_HIGH = 0.05

VALUE_CLIP = 1.0

beta_FULL = 0.10   # SLOW_BIAS_WEIGHT
alpha = 0.05       # SLOW_UPDATE_ALPHA

PERTURB_PATCH = 3  # 3x3
PERTURB_MAG = 0.20

MASTER_SEED = 12345

TAUS = (10, 50, 150)

# Optional features (must default off per spec)
ENABLE_FIG6 = False
ENABLE_FROZEN_SLOW_CONDITION = False


# -----------------------------
# Required functions (spec)
# -----------------------------

def make_dirs() -> None:
    """Create output directories if missing (results/, figures/, runlog/)."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("runlog", exist_ok=True)


def make_run_tag() -> str:
    """Return RUN_TAG formatted as YYYYMMDD_HHMMSS using local time."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_seed_derivation(
    master_seed: int,
    condition: str,
    run_block: str,
    pair_index_within_condition: int,
) -> int:
    """
    Deterministically derive one 32-bit seed from (master_seed, condition, block, pair_index)
    using a stable method that does NOT depend on Python hash randomization.

    Seed formula (allowed example style):
    seed = (MASTER_SEED * 1000003 + crc32("condition|block|pair_index")) mod 2**32
    """
    s = f"{condition}|{run_block}|{pair_index_within_condition}".encode("utf-8")
    crc = zlib.crc32(s) & 0xFFFFFFFF
    seed = (int(master_seed) * 1000003 + int(crc)) % (2**32)
    return int(seed)


def init_states(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize states (float64):
    - fast ~ uniform(-0.05, 0.05)
    - slow = zeros
    """
    fast0 = rng.uniform(FAST_INIT_LOW, FAST_INIT_HIGH, size=(GRID, GRID)).astype(np.float64, copy=False)
    slow0 = np.zeros((GRID, GRID), dtype=np.float64)
    assert fast0.shape == (GRID, GRID)
    assert slow0.shape == (GRID, GRID)
    assert fast0.dtype == np.float64
    assert slow0.dtype == np.float64
    return fast0, slow0


def neighborhood_mean_wrap(arr: np.ndarray) -> np.ndarray:
    """
    Compute neighborhood mean over 9 cells (self + 8 Moore neighbors) with wrap boundary.
    Implementation uses roll-based summation and returns a NEW array (no in-place updates).
    """
    assert arr.shape == (GRID, GRID)
    assert arr.dtype == np.float64

    s = (
        arr +
        np.roll(arr, shift=1, axis=0) +
        np.roll(arr, shift=-1, axis=0) +
        np.roll(arr, shift=1, axis=1) +
        np.roll(arr, shift=-1, axis=1) +
        np.roll(np.roll(arr, shift=1, axis=0), shift=1, axis=1) +
        np.roll(np.roll(arr, shift=1, axis=0), shift=-1, axis=1) +
        np.roll(np.roll(arr, shift=-1, axis=0), shift=1, axis=1) +
        np.roll(np.roll(arr, shift=-1, axis=0), shift=-1, axis=1)
    )
    return s / 9.0


def apply_perturbation_inplace_to_fast_next(fast_next: np.ndarray, sign: float) -> None:
    """
    Apply centered 3x3 perturbation to fast_next IN PLACE (fast only), using wrap boundary.
    Patch is centered at (GRID//2, GRID//2), spans [center-1:center+2] in both axes (with wrap).
    Adds sign * PERTURB_MAG. No extra clipping step is applied afterward (spec).
    """
    assert fast_next.shape == (GRID, GRID)
    assert fast_next.dtype == np.float64
    assert PERTURB_PATCH == 3

    c = GRID // 2  # zero-based
    idx = [((c - 1 + i) % GRID) for i in range(3)]
    fast_next[np.ix_(idx, idx)] += (float(sign) * float(PERTURB_MAG))


def run_one_simulation(
    run_tag: str,
    mode: str,
    condition: str,
    run_block: str,
    pair_id: int,
    pair_index_within_condition: int,
    variant: str,
    seed: int,
    fast_init: np.ndarray,
    slow_init: np.ndarray,
    perturb_sign: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run exactly one simulation (one of A or B for a given pair) and return:
      - metrics_row: dict containing required metadata, sep placeholders, and internal snapshots
      - timeseries_rows: list of per-timestep dict rows for CSV
    """
    # Frozen timing assertions
    assert STEPS == PHASE_WARMUP + PHASE_PERTURB + PHASE_OBSERVE
    t_end = PHASE_WARMUP + PHASE_PERTURB - 1  # 109
    assert t_end == 109
    for tau in TAUS:
        assert 0 <= t_end + tau < STEPS

    # Condition handling (frozen)
    if condition == "full":
        beta = beta_FULL
        use_tanh = True
        frozen_slow = False
    elif condition == "no_feedback":
        beta = 0.0
        use_tanh = True
        frozen_slow = False
    elif condition == "no_nonlinearity":
        beta = beta_FULL
        use_tanh = False  # identity
        frozen_slow = False
    elif condition == "frozen_slow":
        assert ENABLE_FROZEN_SLOW_CONDITION, "frozen_slow condition is disabled by default."
        beta = beta_FULL
        use_tanh = True
        frozen_slow = True
    else:
        raise ValueError(f"Unknown condition: {condition}")

    assert mode in ("quick", "full")
    assert run_block in ("main", "robustness")
    assert variant in ("A", "B")

    # Ensure float64, correct shape, and prevent accidental aliasing of initial arrays
    fast_prev = np.array(fast_init, dtype=np.float64, copy=True)
    slow_prev = np.array(slow_init, dtype=np.float64, copy=True)
    assert fast_prev.shape == (GRID, GRID)
    assert slow_prev.shape == (GRID, GRID)
    assert fast_prev.dtype == np.float64
    assert slow_prev.dtype == np.float64

    capture_times = [t_end + tau for tau in TAUS]
    snapshots: Dict[int, np.ndarray] = {}

    timeseries_rows: List[Dict[str, Any]] = []

    for t in range(STEPS):
        # Fast update (no in-place update to fast_prev/slow_prev)
        avg_fast_prev = neighborhood_mean_wrap(fast_prev)

        if use_tanh:
            fast_next = np.tanh(avg_fast_prev + beta * slow_prev)
        else:
            fast_next = avg_fast_prev + beta * slow_prev

        fast_next = np.clip(fast_next, -VALUE_CLIP, VALUE_CLIP)

        # Self-prediction error (frozen timing): after fast_next, before perturb and before slow_next
        pred_error = float(np.mean(np.abs(slow_prev - fast_next)))

        # Perturbation phase: apply to fast_next only, after fast clip, before slow update
        if PHASE_WARMUP <= t <= (PHASE_WARMUP + PHASE_PERTURB - 1):
            apply_perturbation_inplace_to_fast_next(fast_next, sign=perturb_sign)

        # Global activity uses fast(t) (after perturbation, before slow update)
        global_stat = float(np.mean(np.abs(fast_next)))

        if t in capture_times:
            snapshots[t] = fast_next.copy()

        # Slow update (no in-place update)
        if frozen_slow:
            slow_next = slow_prev.copy()
        else:
            slow_next = (1.0 - alpha) * slow_prev + alpha * fast_next
            slow_next = np.clip(slow_next, -VALUE_CLIP, VALUE_CLIP)

        # Double buffering assertions: new arrays (not aliases of prev)
        assert fast_next is not fast_prev
        assert slow_next is not slow_prev

        timeseries_rows.append({
            "run_tag": run_tag,
            "mode": mode,
            "condition": condition,
            "run_block": run_block,
            "pair_id": int(pair_id),
            "pair_index_within_condition": int(pair_index_within_condition),
            "variant": variant,
            "seed": int(seed),
            "t": int(t),
            "global_stat": global_stat,
            "pred_error": pred_error,
        })

        fast_prev, slow_prev = fast_next, slow_next

    # Ensure we captured all required times
    for ct in capture_times:
        assert ct in snapshots, f"Missing snapshot at t={ct}"

    metrics_row: Dict[str, Any] = {
        "run_tag": run_tag,
        "mode": mode,
        "condition": condition,
        "run_block": run_block,
        "pair_id": int(pair_id),
        "pair_index_within_condition": int(pair_index_within_condition),
        "variant": variant,
        "seed": int(seed),
        # sep placeholders filled after both A and B are run
        "sep_tau10": float("nan"),
        "sep_tau50": float("nan"),
        "sep_tau150": float("nan"),
        # internal (not written to CSV columns):
        "__snapshots": snapshots,
    }

    return metrics_row, timeseries_rows


def run_all(mode: str, run_tag: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Schedule and execute all runs for the selected mode, with one progress bar updating once per run.
    Returns:
      - master_metrics_rows
      - timeseries_rows
      - derived_seeds list for runlog (one entry per simulation run)
    """
    required_conditions = ["full", "no_feedback", "no_nonlinearity"]

    # Build pair schedule: each pair produces two runs (A and B).
    pair_schedule: List[Dict[str, Any]] = []
    next_pair_id = 0

    if mode == "quick":
        # Exactly 1 A/B pair per condition (2 sims each) => 6 simulations total.
        run_block = "main"
        for condition in required_conditions:
            pair_schedule.append({
                "condition": condition,
                "run_block": run_block,
                "pair_id": next_pair_id,
                "pair_index_within_condition": 0,
            })
            next_pair_id += 1
        assert len(pair_schedule) == 3
        assert next_pair_id == 3  # pair_id 0..2
    elif mode == "full":
        # Full mode allocation:
        # Main: 6 pairs per condition (18 pairs)
        # Robustness: 3 pairs per condition (9 pairs)
        # Total pairs = 27 => 54 simulations.
        for run_block, n_pairs in (("main", 6), ("robustness", 3)):
            for condition in required_conditions:
                for pair_index in range(n_pairs):
                    pair_schedule.append({
                        "condition": condition,
                        "run_block": run_block,
                        "pair_id": next_pair_id,
                        "pair_index_within_condition": pair_index,
                    })
                    next_pair_id += 1
        assert len(pair_schedule) == 27
        assert next_pair_id == 27  # pair_id 0..26
    else:
        raise ValueError("mode must be 'quick' or 'full'")

    total_simulations = len(pair_schedule) * 2
    if mode == "quick":
        assert total_simulations == 6
    if mode == "full":
        assert total_simulations == 54

    master_metrics_rows: List[Dict[str, Any]] = []
    timeseries_rows_all: List[Dict[str, Any]] = []
    derived_seeds: List[Dict[str, Any]] = []

    completed_runs = 0

    def progress_bar(done: int, total: int) -> None:
        width = 30
        filled = int(round(width * done / total)) if total > 0 else width
        bar = "#" * filled + "." * (width - filled)
        print(f"\r[{bar}] {done}/{total} runs", end="", flush=True)

    progress_bar(0, total_simulations)

    # Execute each pair: generate init once, run A and B with identical initial arrays, differing only by sign.
    for pair in pair_schedule:
        condition = pair["condition"]
        run_block = pair["run_block"]
        pair_id = int(pair["pair_id"])
        pair_index = int(pair["pair_index_within_condition"])

        seed = stable_seed_derivation(MASTER_SEED, condition, run_block, pair_index)
        rng = np.random.default_rng(seed)
        fast0, slow0 = init_states(rng)

        # Variant A
        mA, tsA = run_one_simulation(
            run_tag=run_tag,
            mode=mode,
            condition=condition,
            run_block=run_block,
            pair_id=pair_id,
            pair_index_within_condition=pair_index,
            variant="A",
            seed=seed,
            fast_init=fast0,  # copied inside
            slow_init=slow0,  # copied inside
            perturb_sign=+1.0,
        )
        completed_runs += 1
        progress_bar(completed_runs, total_simulations)

        # Variant B (must reuse exact initial arrays from A, copied; not regenerated)
        mB, tsB = run_one_simulation(
            run_tag=run_tag,
            mode=mode,
            condition=condition,
            run_block=run_block,
            pair_id=pair_id,
            pair_index_within_condition=pair_index,
            variant="B",
            seed=seed,
            fast_init=fast0,  # copied inside; no regeneration
            slow_init=slow0,  # copied inside; no regeneration
            perturb_sign=-1.0,
        )
        completed_runs += 1
        progress_bar(completed_runs, total_simulations)

        # Compute frozen separation metrics from snapshots (fast only) at t_end+tau
        snapsA: Dict[int, np.ndarray] = mA["__snapshots"]
        snapsB: Dict[int, np.ndarray] = mB["__snapshots"]
        t_end = PHASE_WARMUP + PHASE_PERTURB - 1

        sep_values: Dict[int, float] = {}
        for tau in TAUS:
            t_sep = t_end + tau
            a = snapsA[t_sep]
            b = snapsB[t_sep]
            sep = float(np.mean(np.abs(a - b)))
            sep_values[tau] = sep

        mA["sep_tau10"] = sep_values[10]
        mA["sep_tau50"] = sep_values[50]
        mA["sep_tau150"] = sep_values[150]
        mB["sep_tau10"] = sep_values[10]
        mB["sep_tau50"] = sep_values[50]
        mB["sep_tau150"] = sep_values[150]

        # Remove internal snapshots before CSV writing (write_csvs ignores extras, but keep clean)
        mA.pop("__snapshots", None)
        mB.pop("__snapshots", None)

        master_metrics_rows.extend([mA, mB])
        timeseries_rows_all.extend(tsA)
        timeseries_rows_all.extend(tsB)

        derived_seeds.append({
            "condition": condition,
            "run_block": run_block,
            "pair_id": pair_id,
            "pair_index_within_condition": pair_index,
            "variant": "A",
            "seed": int(seed),
        })
        derived_seeds.append({
            "condition": condition,
            "run_block": run_block,
            "pair_id": pair_id,
            "pair_index_within_condition": pair_index,
            "variant": "B",
            "seed": int(seed),
        })

    print()  # newline after progress bar

    # Final run-count assertions
    assert len(master_metrics_rows) == total_simulations
    assert len(timeseries_rows_all) == total_simulations * STEPS

    return master_metrics_rows, timeseries_rows_all, derived_seeds


def write_csvs(
    run_tag: str,
    master_metrics_rows: List[Dict[str, Any]],
    timeseries_rows: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Write the required CSVs with EXACT columns and order.
    Returns (metrics_csv_path, timeseries_csv_path).
    """
    metrics_path = os.path.join("results", f"master_metrics_{run_tag}.csv")
    timeseries_path = os.path.join("results", f"timeseries_{run_tag}.csv")

    metrics_cols = [
        "run_tag",
        "mode",
        "condition",
        "run_block",
        "pair_id",
        "pair_index_within_condition",
        "variant",
        "seed",
        "sep_tau10",
        "sep_tau50",
        "sep_tau150",
    ]

    timeseries_cols = [
        "run_tag",
        "mode",
        "condition",
        "run_block",
        "pair_id",
        "pair_index_within_condition",
        "variant",
        "seed",
        "t",
        "global_stat",
        "pred_error",
    ]

    # master_metrics CSV
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(metrics_cols)
        for row in master_metrics_rows:
            w.writerow([row[c] for c in metrics_cols])

    # timeseries CSV
    with open(timeseries_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(timeseries_cols)
        for row in timeseries_rows:
            w.writerow([row[c] for c in timeseries_cols])

    return metrics_path, timeseries_path


def write_runlog(
    run_tag: str,
    mode: str,
    derived_seeds: List[Dict[str, Any]],
) -> str:
    """
    Write runlog/run_<RUN_TAG>.json with required fields.
    """
    t_end = PHASE_WARMUP + PHASE_PERTURB - 1

    payload = {
        "run_tag": run_tag,
        "mode": mode,
        "master_seed": MASTER_SEED,
        "derived_seeds": derived_seeds,
        "frozen_parameters": {
            "GRID": GRID,
            "STEPS": STEPS,
            "PHASE_WARMUP": PHASE_WARMUP,
            "PHASE_PERTURB": PHASE_PERTURB,
            "PHASE_OBSERVE": PHASE_OBSERVE,
            "VALUE_CLIP": VALUE_CLIP,
            "alpha": alpha,
            "beta": beta_FULL,
            "PERTURB_PATCH": PERTURB_PATCH,
            "PERTURB_MAG": PERTURB_MAG,
            "BOUNDARY": BOUNDARY,
        },
        "time_indexing_statement": {
            "t_definition": "t = 0 is the first update step after initialization",
            "warmup": "t = 0..99",
            "perturb": "t = 100..109",
            "observe": "t = 110..359",
            "t_end": int(t_end),
            "separation_evaluation": {str(tau): int(t_end + tau) for tau in TAUS},
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "timestamp": _dt.datetime.now().isoformat(),
    }

    path = os.path.join("runlog", f"run_{run_tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    return path


def generate_figures_from_csvs(
    run_tag: str,
    metrics_csv_path: str,
    timeseries_csv_path: str,
) -> None:
    """
    Generate figures using ONLY the CSV files that were written.
    Saves:
      - fig1_model_schematic_<RUN_TAG>.png
      - fig2_time_evolution_<RUN_TAG>.png
      - fig3_history_dependence_<RUN_TAG>.png
      - fig4_ablation_collapse_<RUN_TAG>.png
      - fig5_self_prediction_<RUN_TAG>.png
      - fig6_robustness_<RUN_TAG>.png (optional; only if ENABLE_FIG6=True)
    """

    # --- Read master_metrics CSV ---
    metrics_rows: List[Dict[str, str]] = []
    with open(metrics_csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            metrics_rows.append(row)

    # --- Read timeseries CSV ---
    ts_rows: List[Dict[str, str]] = []
    with open(timeseries_csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts_rows.append(row)

    # Helper: parse float safely
    def ffloat(x: str) -> float:
        return float(x)

    # Helper: group timeseries means by (condition, metric) over t
    def mean_timeseries_by_condition(metric_name: str, conditions: List[str]) -> Dict[str, np.ndarray]:
        sums = {c: np.zeros(STEPS, dtype=np.float64) for c in conditions}
        counts = {c: np.zeros(STEPS, dtype=np.int64) for c in conditions}

        for row in ts_rows:
            c = row["condition"]
            if c not in sums:
                continue
            t = int(row["t"])
            sums[c][t] += ffloat(row[metric_name])
            counts[c][t] += 1

        means: Dict[str, np.ndarray] = {}
        for c in conditions:
            assert np.all(counts[c] > 0), f"Missing timeseries rows for condition={c}"
            means[c] = sums[c] / counts[c]
        return means

    # Helper: mean separations by condition from master metrics
    def mean_seps_by_condition(conditions: List[str]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for c in conditions:
            vals10: List[float] = []
            vals50: List[float] = []
            vals150: List[float] = []
            for row in metrics_rows:
                if row["condition"] != c:
                    continue
                vals10.append(ffloat(row["sep_tau10"]))
                vals50.append(ffloat(row["sep_tau50"]))
                vals150.append(ffloat(row["sep_tau150"]))
            assert len(vals10) > 0, f"No metrics rows for condition={c}"
            out[c] = {
                "tau10": float(np.mean(vals10)),
                "tau50": float(np.mean(vals50)),
                "tau150": float(np.mean(vals150)),
            }
        return out

    # ---------------- Figure 1: schematic ----------------
    fig1_path = os.path.join("figures", f"fig1_model_schematic_{run_tag}.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axis_off()
    ax.set_title("Figure 1 — Model schematic (frozen)", pad=14)

    # Stable coordinate system (0..1 in both axes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    tr = ax.transAxes

    # Boxes (spaced to avoid overlaps)
    grid_box = Rectangle((0.06, 0.56), 0.26, 0.30, fill=False, linewidth=2.5, transform=tr)
    fast_box = Rectangle((0.42, 0.66), 0.18, 0.16, fill=False, linewidth=2.5, transform=tr)
    slow_box = Rectangle((0.42, 0.38), 0.18, 0.16, fill=False, linewidth=2.5, transform=tr)

    ax.add_patch(grid_box)
    ax.add_patch(fast_box)
    ax.add_patch(slow_box)

    # Box labels
    ax.text(
        0.19, 0.71, "2D grid\n(32×32)\nwrap boundary",
        ha="center", va="center", fontsize=12, transform=tr
    )
    ax.text(0.51, 0.74, "fast state", ha="center", va="center", fontsize=12, transform=tr)
    ax.text(0.51, 0.46, "slow state", ha="center", va="center", fontsize=12, transform=tr)

    # Arrow: grid -> fast (Moore neighborhood mean)
    ax.annotate(
        "",
        xy=(0.42, 0.74),
        xytext=(0.32, 0.74),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2.5),
    )
    # Place the Moore label ABOVE the fast box so it never overlaps
    ax.text(
        0.33, 0.86,
        "Moore neighborhood mean\n(self + 8 neighbors)",
        ha="left", va="center", fontsize=11, transform=tr,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
    )

    # Arrow: fast -> slow (slow update uses fast_next)
    ax.annotate(
        "",
        xy=(0.51, 0.54),
        xytext=(0.51, 0.66),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2.5),
    )

    # Arrow: slow -> fast (feedback via beta * slow_prev)
    # Route it LEFT of both boxes (negative curvature), so it doesn't cut through them.
    ax.annotate(
        "",
        xy=(0.41, 0.72),       # just left of fast box
        xytext=(0.41, 0.46),   # just left of slow box
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2.5, connectionstyle="arc3,rad=-0.8"),
    )
    # Put feedback label away from the boxes
    ax.text(
        0.20, 0.46,
        "feedback via β·slow_prev",
        ha="left", va="center", fontsize=11, transform=tr,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
    )

    # Right-side formula boxes (separate from arrows/boxes)
    ax.text(
        0.66, 0.78,
        "fast update:\nfast_next = tanh(avg_fast_prev + β·slow_prev)\n(or identity ablation)",
        ha="left", va="center", fontsize=11, transform=tr,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1.5),
    )
    ax.text(
        0.66, 0.47,
        "slow update:\nslow_next = (1-α)·slow_prev + α·fast_next",
        ha="left", va="center", fontsize=11, transform=tr,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1.5),
    )

    # Required exclusion statement (big + visible)
    ax.text(
        0.06, 0.10,
        "Explicitly excluded:\nno goals, no learning, no environment, no policies",
        ha="left", va="bottom", fontsize=14, transform=tr,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", lw=2.0),
    )

    fig.savefig(fig1_path, dpi=200)
    plt.close(fig)

    # ---------------- Figure 2: time evolution (full) ----------------
    fig2_path = os.path.join("figures", f"fig2_time_evolution_{run_tag}.png")

    # mean global_stat(t) for condition="full" across all full runs (A/B, main/robustness) from CSV
    sums = np.zeros(STEPS, dtype=np.float64)
    counts = np.zeros(STEPS, dtype=np.int64)
    for row in ts_rows:
        if row["condition"] != "full":
            continue
        t = int(row["t"])
        sums[t] += ffloat(row["global_stat"])
        counts[t] += 1
    assert np.all(counts > 0), "No full-condition timeseries rows found."
    mean_global = sums / counts

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(STEPS), mean_global)
    ax.set_title("Figure 2 — Time evolution (full model): mean global activity")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("global_stat(t) = mean(|fast(t)|)")

    # phase markers
    ax.axvline(PHASE_WARMUP, linestyle="--")
    ax.axvline(PHASE_WARMUP + PHASE_PERTURB, linestyle="--")
    ax.text(PHASE_WARMUP / 2, ax.get_ylim()[1] * 0.95, "warmup", ha="center", va="top")
    ax.text(PHASE_WARMUP + PHASE_PERTURB / 2, ax.get_ylim()[1] * 0.95, "perturb", ha="center", va="top")
    ax.text(PHASE_WARMUP + PHASE_PERTURB + PHASE_OBSERVE / 2, ax.get_ylim()[1] * 0.95, "observe", ha="center", va="top")

    fig.tight_layout()
    fig.savefig(fig2_path, dpi=200)
    plt.close(fig)

    # ---------------- Figure 3: history dependence (grouped bars) ----------------
    fig3_path = os.path.join("figures", f"fig3_history_dependence_{run_tag}.png")

    conditions = ["full", "no_feedback", "no_nonlinearity"]
    seps = mean_seps_by_condition(conditions)

    taus = [10, 50, 150]
    x = np.arange(len(conditions), dtype=np.float64)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width, [seps[c]["tau10"] for c in conditions], width, label="τ=10")
    ax.bar(x,         [seps[c]["tau50"] for c in conditions], width, label="τ=50")
    ax.bar(x + width, [seps[c]["tau150"] for c in conditions], width, label="τ=150")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("A/B separation = mean(|fast_A - fast_B|)")
    ax.set_title("Figure 3 — History dependence (mean separation across runs)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig3_path, dpi=200)
    plt.close(fig)

    # ---------------- Figure 4: ablation collapse modes ----------------
    fig4_path = os.path.join("figures", f"fig4_ablation_collapse_{run_tag}.png")

    ablations = ["no_feedback", "no_nonlinearity"]
    seps_ab = mean_seps_by_condition(ablations)
    x2 = np.arange(len(ablations), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x2 - width, [seps_ab[c]["tau10"] for c in ablations], width, label="τ=10")
    ax.bar(x2,         [seps_ab[c]["tau50"] for c in ablations], width, label="τ=50")
    ax.bar(x2 + width, [seps_ab[c]["tau150"] for c in ablations], width, label="τ=150")
    ax.set_xticks(x2)
    ax.set_xticklabels(ablations)
    ax.set_ylabel("A/B separation = mean(|fast_A - fast_B|)")
    ax.set_title("Figure 4 — Ablation collapse modes (mean separation)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig4_path, dpi=200)
    plt.close(fig)

    # ---------------- Figure 5: self-prediction consistency ----------------
    fig5_path = os.path.join("figures", f"fig5_self_prediction_{run_tag}.png")

    pred_means = mean_timeseries_by_condition("pred_error", ["full", "no_feedback", "no_nonlinearity"])

    fig, ax = plt.subplots(figsize=(9, 4))
    t = np.arange(STEPS)
    ax.plot(t, pred_means["full"], label="full")
    ax.plot(t, pred_means["no_feedback"], label="no_feedback")
    ax.plot(t, pred_means["no_nonlinearity"], label="no_nonlinearity")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("pred_error(t) = mean(|slow_prev - fast_next|)")
    ax.set_title("Figure 5 — Self-prediction consistency (mean pred_error across runs)")
    ax.legend()

    # phase markers
    ax.axvline(PHASE_WARMUP, linestyle="--")
    ax.axvline(PHASE_WARMUP + PHASE_PERTURB, linestyle="--")

    fig.tight_layout()
    fig.savefig(fig5_path, dpi=200)
    plt.close(fig)

    # ---------------- Figure 6: robustness (optional) ----------------
    if ENABLE_FIG6:
        fig6_path = os.path.join("figures", f"fig6_robustness_{run_tag}.png")

        # Show variability across robustness seeds using existing separation metric sep_tau150
        # (no new metrics invented). Use only master_metrics CSV.
        conditions6 = ["full", "no_feedback", "no_nonlinearity"]
        data: Dict[str, List[float]] = {c: [] for c in conditions6}
        for row in metrics_rows:
            if row["run_block"] != "robustness":
                continue
            c = row["condition"]
            if c in data:
                data[c].append(ffloat(row["sep_tau150"]))

        # It's valid for quick mode to have no robustness block; still implement path.
        if any(len(v) > 0 for v in data.values()):
            fig, ax = plt.subplots(figsize=(9, 4))
            positions = np.arange(len(conditions6))
            for i, c in enumerate(conditions6):
                y = data[c]
                if len(y) == 0:
                    continue
                ax.scatter(np.full(len(y), positions[i], dtype=np.float64), y, label=c)
            ax.set_xticks(positions)
            ax.set_xticklabels(conditions6)
            ax.set_ylabel("sep_tau150")
            ax.set_title("Figure 6 — Robustness (robustness block variability; sep_tau150)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig6_path, dpi=200)
            plt.close(fig)


def main() -> None:
    """Entry point with mode prompt."""
    make_dirs()
    run_tag = make_run_tag()

    print("[1] Quick mode — pipeline check")
    print("[2] Full mode — paper results")
    choice = input("Select mode (1 or 2): ").strip()

    while choice not in ("1", "2"):
        choice = input("Please enter 1 (Quick) or 2 (Full): ").strip()

    mode = "quick" if choice == "1" else "full"

    # Run simulations
    master_metrics_rows, timeseries_rows, derived_seeds = run_all(mode=mode, run_tag=run_tag)

    # Write CSVs
    metrics_csv_path, timeseries_csv_path = write_csvs(
        run_tag=run_tag,
        master_metrics_rows=master_metrics_rows,
        timeseries_rows=timeseries_rows,
    )

    # Write runlog
    runlog_path = write_runlog(run_tag=run_tag, mode=mode, derived_seeds=derived_seeds)

    # Generate figures from CSV only
    generate_figures_from_csvs(run_tag=run_tag, metrics_csv_path=metrics_csv_path, timeseries_csv_path=timeseries_csv_path)

    # Minimal end-of-run summary (no per-timestep output)
    print("Done.")
    print(f"RUN_TAG: {run_tag}")
    print(f"Wrote: {metrics_csv_path}")
    print(f"Wrote: {timeseries_csv_path}")
    print(f"Wrote: {runlog_path}")
    print("Wrote: figures/fig1..fig5 PNGs (fig6 optional, default disabled)")


if __name__ == "__main__":
    main()
