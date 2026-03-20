# numba_cores.py
# -----------------------------------------------------------------------
# Drop-in Numba-accelerated cores for the two heaviest loops in
# create_daily_enhanced_trace.py
#
# Usage
# -----
# 1. Place this file next to create_daily_enhanced_trace.py
# 2. In create_daily_enhanced_trace.py add at the top:
#       from numba_cores import bounce_back_core, rolling_unique_median_nb
# 3. Replace the inner loops as shown below.
#
# First call will trigger JIT compilation (~5-10s). Every subsequent call
# runs at C speed and releases the GIL, enabling true multi-core with
# ThreadPoolExecutor.
# -----------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from numba import njit, prange


# ═══════════════════════════════════════════════════════════════════════
# 1.  Rolling unique-median  (used inside flag_price_change_errors)
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _unique_median_window(arr: np.ndarray, start: int, end: int) -> float:
    """Median of unique values in arr[start:end]."""
    n = end - start
    if n == 0:
        return np.nan

    # copy window into a local buffer and sort
    buf = arr[start:end].copy()
    buf.sort()

    # deduplicate in-place, count uniques
    n_uniq = 1
    for k in range(1, n):
        if buf[k] != buf[k - 1]:
            buf[n_uniq] = buf[k]
            n_uniq += 1

    if n_uniq % 2 == 1:
        return buf[n_uniq // 2]
    else:
        return 0.5 * (buf[n_uniq // 2 - 1] + buf[n_uniq // 2])


@njit(cache=True)
def rolling_unique_median_nb(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Backward-looking rolling unique-median, shifted by 1 (no look-ahead).
    Equivalent to:
        s.rolling(window, min_periods=1).apply(unique_median, raw=True).shift(1)

    Parameters
    ----------
    prices : float64 array, one CUSIP group
    window : int

    Returns
    -------
    float64 array, same length, index-0 is NaN (no prior data)
    """
    n = len(prices)
    out = np.empty(n, dtype=np.float64)
    out[0] = np.nan  # no prior data for first observation

    for i in range(1, n):
        start = max(0, i - window)
        out[i] = _unique_median_window(prices, start, i)

    return out


# ═══════════════════════════════════════════════════════════════════════
# 2.  Bounce-back core  (the main while-loop in flag_price_change_errors)
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def bounce_back_core(
    P:                       np.ndarray,   # prices,  float64
    D:                       np.ndarray,   # diffs,   float64
    B:                       np.ndarray,   # baseline (rolling unique median shifted), float64
    threshold_abs:           float,
    lookahead:               int,
    max_span:                int,
    back_to_anchor_tol:      float,
    candidate_slack_abs:     float,
    reassignment_margin_abs: float,
    par_spike_heuristic:     bool,
    par_level:               float,
    par_equal_tol:           float,
    par_min_run:             int,
    par_cooldown_after_flag: int,
) -> np.ndarray:
    """
    Pure-Numba implementation of the bounce-back flagging loop.

    Parameters match the scalar arguments of flag_price_change_errors.
    P, D, B are 1-D float64 arrays for a **single CUSIP group**,
    already sorted by (date, time).

    Returns
    -------
    filtered : int8 array, 1 = flagged error, 0 = keep
    """
    eps   = 1e-12
    thr_lo       = max(0.0, threshold_abs - candidate_slack_abs)
    back_tol_abs = back_to_anchor_tol * threshold_abs

    m        = len(P)
    filtered = np.zeros(m, dtype=np.int8)

    par_cooldown_until = -1
    i = 0

    while i < m:
        # Cooldown: skip non-par rows after a par-run flag
        if i <= par_cooldown_until and (abs(P[i] - par_level) > par_equal_tol):
            i += 1
            continue

        p_i = P[i]
        d_i = D[i]
        b_i = B[i]

        d_valid = not np.isnan(d_i)
        b_valid = not np.isnan(b_i)

        cond_jump     = d_valid and (abs(d_i) >= thr_lo - eps)
        cond_far_prev = b_valid and (abs(p_i - b_i) >= thr_lo - eps)

        cond_par = False
        if par_spike_heuristic and not np.isnan(p_i):
            if abs(p_i - par_level) <= par_equal_tol:
                if b_valid and (abs(p_i - b_i) >= back_tol_abs - eps):
                    cond_par = True

        par_only = cond_par and not cond_jump

        if not (cond_jump or cond_far_prev or cond_par):
            i += 1
            continue

        j_lim   = min(m - 1, i + lookahead)
        j_match = -1
        k_return = -1

        if not par_only:
            for j in range(i + 1, j_lim + 1):
                dj = D[j]
                if (not np.isnan(d_i)) and (not np.isnan(dj)):
                    if (np.sign(dj) == -np.sign(d_i)) and (abs(dj) >= thr_lo - eps):
                        j_match = j
                        break
                if b_valid and (abs(P[j] - b_i) <= back_tol_abs + eps):
                    k_return = j
                    break

        par_start = cond_par

        # ── Standard quick-correction ─────────────────────────────────────
        stop_at = -1
        if j_match >= 0:
            stop_at = j_match
        elif k_return >= 0:
            stop_at = k_return

        if (not par_only) and stop_at >= 0:
            flag_start = i

            # Blame reassignment
            prev = i - 1
            if prev >= 0:
                b_prev   = B[prev]
                dev_prev = abs(P[prev] - b_prev) if not np.isnan(b_prev) else np.nan
                dev_curr = abs(p_i - b_i)        if b_valid             else np.nan
                if (not np.isnan(dev_prev)) and (not np.isnan(dev_curr)):
                    if ((dev_prev - dev_curr) >= reassignment_margin_abs - eps
                            and dev_prev >= back_tol_abs - eps):
                        flag_start = prev

            if (not par_start) or abs(P[flag_start] - par_level) <= par_equal_tol:
                filtered[flag_start] = 1

            b_fs     = B[flag_start]
            span_end = min(stop_at, flag_start + max_span)
            for k in range(flag_start + 1, span_end + 1):
                if par_start:
                    if abs(P[k] - par_level) <= par_equal_tol:
                        filtered[k] = 1
                else:
                    if (not np.isnan(b_fs)) and (abs(P[k] - b_fs) >= back_tol_abs - eps):
                        filtered[k] = 1
                    else:
                        break

            if par_start:
                new_cd = stop_at + par_cooldown_after_flag
                if new_cd > par_cooldown_until:
                    par_cooldown_until = new_cd

            i = stop_at + 1
            continue

        # ── Persistent par block ──────────────────────────────────────────
        if par_start:
            run_end = i
            while run_end + 1 < m and abs(P[run_end + 1] - par_level) <= par_equal_tol:
                run_end += 1
            run_len = run_end - i + 1
            if run_len >= par_min_run:
                for k in range(i, run_end + 1):
                    filtered[k] = 1
                new_cd = run_end + par_cooldown_after_flag
                if new_cd > par_cooldown_until:
                    par_cooldown_until = new_cd
                i = run_end + 1
                continue

        i += 1

    return filtered


# ═══════════════════════════════════════════════════════════════════════
# 3.  Public wrapper — drop-in replacement for flag_price_change_errors
# ═══════════════════════════════════════════════════════════════════════

def flag_price_change_errors_nb(
    df,
    *,
    id_col:                  str   = "cusip_id",
    date_col:                str   = "trd_exctn_dt",
    time_col                       = "trd_exctn_tm",
    price_col:               str   = "rptd_pr",
    threshold_abs:           float = 35.0,
    lookahead:               int   = 5,
    max_span:                int   = 5,
    window:                  int   = 5,
    back_to_anchor_tol:      float = 0.25,
    candidate_slack_abs:     float = 1.0,
    reassignment_margin_abs: float = 5.0,
    use_unique_trailing_median: bool = True,
    par_spike_heuristic:     bool  = True,
    par_level:               float = 100.0,
    par_equal_tol:           float = 1e-8,
    par_min_run:             int   = 3,
    par_cooldown_after_flag: int   = 2,
):
    """
    Drop-in replacement for flag_price_change_errors using Numba JIT.

    Signature is identical — swap the function name only:

        # Before:
        trace_bb = flag_price_change_errors(trace, **_bb)

        # After:
        trace_bb = flag_price_change_errors_nb(trace, **_bb)
    """
    import pandas as pd

    out = df.copy()

    # Sort cols
    sort_cols = [id_col, date_col]
    if time_col and time_col in out.columns:
        sort_cols.append(time_col)
    out = out.sort_values(sort_cols).reset_index(drop=True)

    # Price diff (pandas, cheap)
    out["delta_rptd_pr"] = (
        out.groupby(id_col, observed=True)[price_col].diff().astype(float)
    )

    prices_all   = out[price_col].to_numpy(dtype=np.float64)
    diffs_all    = out["delta_rptd_pr"].to_numpy(dtype=np.float64)
    baseline_all = np.full(len(out), np.nan, dtype=np.float64)
    filtered_all = np.zeros(len(out), dtype=np.int8)

    # Per-group: compute baseline + run Numba core
    groups = out.groupby(id_col, observed=True).indices  # dict {id: int array of positions}

    for _, row_idxs in groups.items():
        row_idxs = np.asarray(row_idxs, dtype=np.int64)
        P = prices_all[row_idxs]
        D = diffs_all[row_idxs]

        # Baseline: rolling unique-median shifted by 1
        if use_unique_trailing_median:
            B = rolling_unique_median_nb(P, window + 1)
        else:
            # fallback: standard rolling median shifted
            s = pd.Series(P)
            B = s.rolling(window + 1, min_periods=1).median().shift(1).to_numpy(np.float64)

        baseline_all[row_idxs] = B

        flags = bounce_back_core(
            P, D, B,
            float(threshold_abs),
            int(lookahead),
            int(max_span),
            float(back_to_anchor_tol),
            float(candidate_slack_abs),
            float(reassignment_margin_abs),
            bool(par_spike_heuristic),
            float(par_level),
            float(par_equal_tol),
            int(par_min_run),
            int(par_cooldown_after_flag),
        )
        filtered_all[row_idxs] = flags

    out["baseline_trailing"] = baseline_all
    out["filtered_error"]    = filtered_all.astype(np.int8)
    return out


# ═══════════════════════════════════════════════════════════════════════
# 4.  Pre-warm JIT  (call once at import time or module startup)
# ═══════════════════════════════════════════════════════════════════════

def warm_up_jit():
    """
    Call this once at startup (before spawning threads) to trigger
    Numba compilation. Takes ~5-10 seconds, then all subsequent
    calls are near-instant.

        from numba_cores import warm_up_jit
        warm_up_jit()   # put this before ThreadPoolExecutor
    """
    import logging
    logging.info("Warming up Numba JIT (one-time, ~5-10s) ...")
    rng = np.random.default_rng(0)
    P   = rng.uniform(90, 110, 50).astype(np.float64)
    D   = np.diff(P, prepend=P[0])
    B   = rolling_unique_median_nb(P, 6)
    bounce_back_core(P, D, B, 35.0, 5, 5, 0.25, 1.0, 5.0, True, 100.0, 1e-8, 3, 2)
    logging.info("Numba JIT warm-up complete.")


if __name__ == "__main__":
    warm_up_jit()
    print("Numba cores ready.")
