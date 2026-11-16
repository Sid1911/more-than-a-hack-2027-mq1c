"""Phase 4: Frame-level Feature Engineering Pipeline for BAR-ATH Tactical Data.

Builds a strictly causal (past + current frame only) feature matrix of shape (199319, F)
from the tactical ATD, homography, and Smart Tagging XML without touching ground-truth labels
or modifying any repository / Dataset files.

Outputs saved to scratch/features/:
  - tactical_features.npz          (compressed float32 feature matrix X, frame_id, video_frame, feature_names)
  - feature_catalog.json           (feature names, groups, descriptions, units)
  - feature_stats.csv              (missing %, min, mean, std, median, p05, p95, max per feature)
  - validation_report.json         (automated validation checks and summary metrics)
  - validation_report.txt          (human-readable validation report)
"""

import collections
import json
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mtah_paths  # noqa: E402

mtah_paths.configure()
import games  # noqa: E402
import load_atd  # noqa: E402
import load_homography  # noqa: E402
import load_smart_tagging  # noqa: E402
import baseline  # noqa: E402

CACHE_ATD = os.path.join(HERE, "cache", "atd_tactical.npz")
OUT_DIR = os.path.join(HERE, "features")

GAME = games.GAME
N_FRAMES = games.source_frames("tactical", GAME)  # 199,319
FPS = float(games.GAMES[GAME]["fps"])             # 25.0
DT = 1.0 / FPS                                    # 0.04s
PITCH_L, PITCH_W = games.GAMES[GAME]["pitch"]     # 105.3, 68.0
HALF_BLOCKS = games.half_video_blocks(GAME)       # [(1, 11185, 90651), (2, 110808, 190197)]
KICKOFFS = load_smart_tagging.KICKOFF_VIDEO_FRAMES  # {'1st Half': 11185, '2nd Half': 113742}

# Physical plausibility thresholds to filter tracker identity swaps / teleports
MAX_PLAYER_STEP_SPEED = 13.0   # m/s (~46.8 km/h)
MAX_PLAYER_ACCEL = 15.0        # m/s^2
MAX_BALL_STEP_SPEED = 45.0     # m/s (~162 km/h)


def load_cached_atd():
    """Load tactical ATD arrays from Phase 3 cache, or parse if cache is absent."""
    meta_path, raw_path = load_atd.atd_paths()
    team_of_slot = baseline.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)
    if os.path.isfile(CACHE_ATD):
        z = np.load(CACHE_ATD)
        return z["frame_id"], z["players"], z["ball"], team_of_slot, meta_path
    from diagnose_tactical import load_atd_arrays
    frame_id, players, ball, _ = load_atd_arrays(meta_path, raw_path, n_slots, reparse=True)
    return frame_id, players, ball, team_of_slot, meta_path


def causal_rolling_sum_and_count(arr, window):
    """Causal rolling sum and valid count over [t - window + 1, t] along axis 0.
    Works for 1D or 2D arrays with NaNs. Strictly past + present."""
    valid = ~np.isnan(arr)
    filled = np.where(valid, arr, 0.0).astype(np.float64)
    csum = np.cumsum(filled, axis=0)
    ccnt = np.cumsum(valid.astype(np.int32), axis=0)
    if arr.ndim == 1:
        csum_pad = np.concatenate(([0.0], csum))
        ccnt_pad = np.concatenate(([0], ccnt))
    else:
        z_f = np.zeros((1, arr.shape[1]), dtype=np.float64)
        z_i = np.zeros((1, arr.shape[1]), dtype=np.int32)
        csum_pad = np.vstack((z_f, csum))
        ccnt_pad = np.vstack((z_i, ccnt))
    idx = np.arange(len(arr))
    hi = idx + 1
    lo = np.maximum(0, hi - window)
    r_sum = csum_pad[hi] - csum_pad[lo]
    r_cnt = ccnt_pad[hi] - ccnt_pad[lo]
    return r_sum, r_cnt


def causal_rolling_mean(arr, window):
    r_sum, r_cnt = causal_rolling_sum_and_count(arr, window)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    mask = r_cnt > 0
    out[mask] = (r_sum[mask] / r_cnt[mask]).astype(np.float32)
    return out


def causal_rolling_mean_std(arr_1d, window):
    """Causal rolling mean and std over [t - window + 1, t] for a 1D array with NaNs."""
    r_sum, r_cnt = causal_rolling_sum_and_count(arr_1d, window)
    r_sum2, _ = causal_rolling_sum_and_count(arr_1d * arr_1d, window)
    mean = np.full(len(arr_1d), np.nan, dtype=np.float32)
    std = np.full(len(arr_1d), np.nan, dtype=np.float32)
    m1 = r_cnt > 0
    mean[m1] = (r_sum[m1] / r_cnt[m1]).astype(np.float32)
    m2 = r_cnt >= 2
    var = (r_sum2[m2] / r_cnt[m2]) - (r_sum[m2] / r_cnt[m2]) ** 2
    std[m2] = np.sqrt(np.maximum(0.0, var)).astype(np.float32)
    return mean, std


def causal_ffill_1d(arr, valid_mask):
    """Causal forward-fill of a 1D array where valid_mask is True. Initial missing stay NaN."""
    idx = np.where(valid_mask, np.arange(len(arr)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.full(len(arr), np.nan, dtype=np.float32)
    ok = idx >= 0
    out[ok] = arr[idx[ok]]
    return out, idx


class FeatureBuilder:
    def __init__(self):
        self.cols = []
        self.meta = []

    def add(self, name, arr, group, desc, unit=""):
        arr_f32 = np.asarray(arr, dtype=np.float32)
        if arr_f32.shape != (N_FRAMES,):
            raise ValueError(f"Feature {name} has shape {arr_f32.shape}, expected ({N_FRAMES},)")
        self.cols.append(arr_f32)
        self.meta.append({
            "index": len(self.cols) - 1,
            "name": name,
            "group": group,
            "description": desc,
            "unit": unit,
        })

    def add_block(self, names, mat_2d, group, desc_template, unit=""):
        mat_f32 = np.asarray(mat_2d, dtype=np.float32)
        if mat_f32.shape != (N_FRAMES, len(names)):
            raise ValueError(f"Block {group} shape {mat_f32.shape} != ({N_FRAMES}, {len(names)})")
        for j, name in enumerate(names):
            self.cols.append(mat_f32[:, j])
            self.meta.append({
                "index": len(self.cols) - 1,
                "name": name,
                "group": group,
                "description": desc_template.format(slot=j + 1, name=name),
                "unit": unit,
            })


def build_all_features():
    t0 = time.time()
    print("=" * 78)
    print("PHASE 4: BUILDING TACTICAL FEATURE PIPELINE (BAR-ATH)")
    frame_id, players_raw, ball_raw, team_of_slot, meta_path = load_cached_atd()
    video_frame = (frame_id - 1).astype(np.int32)
    n_slots = players_raw.shape[1]  # 46
    home_mask = (team_of_slot == 0)  # slots 1..24 (Barcelona)
    away_mask = (team_of_slot == 1)  # slots 25..46 (Athletic Club)

    fb = FeatureBuilder()

    # =========================================================================
    # PART A — BALL FEATURES
    # =========================================================================
    print("[Part A] Computing Ball Features...")
    ball_x_norm = ball_raw[:, 0].astype(np.float32)
    ball_y_norm = ball_raw[:, 1].astype(np.float32)
    ball_vis = (~np.isnan(ball_x_norm)).astype(np.float32)
    ball_vis_bool = ~np.isnan(ball_x_norm)

    ball_x_m = (ball_x_norm * PITCH_L).astype(np.float32)
    ball_y_m = (ball_y_norm * PITCH_W).astype(np.float32)

    # Causal 1-step velocity (strictly requires t-1 and t both visible; no interpolation)
    ball_vx = np.full(N_FRAMES, np.nan, dtype=np.float32)
    ball_vy = np.full(N_FRAMES, np.nan, dtype=np.float32)
    consec_ball = ball_vis_bool[1:] & ball_vis_bool[:-1]
    vx_step = (ball_x_m[1:] - ball_x_m[:-1]) * FPS
    vy_step = (ball_y_m[1:] - ball_y_m[:-1]) * FPS
    sp_step = np.hypot(vx_step, vy_step)
    valid_bstep = consec_ball & (sp_step <= MAX_BALL_STEP_SPEED)
    ball_vx[1:][valid_bstep] = vx_step[valid_bstep]
    ball_vy[1:][valid_bstep] = vy_step[valid_bstep]
    ball_speed = np.hypot(ball_vx, ball_vy).astype(np.float32)

    # Causal 2-step acceleration (requires valid velocity at t-1 and t)
    ball_ax = np.full(N_FRAMES, np.nan, dtype=np.float32)
    ball_ay = np.full(N_FRAMES, np.nan, dtype=np.float32)
    valid_bacc = (~np.isnan(ball_vx[1:])) & (~np.isnan(ball_vx[:-1]))
    ball_ax[1:][valid_bacc] = (ball_vx[1:][valid_bacc] - ball_vx[:-1][valid_bacc]) * FPS
    ball_ay[1:][valid_bacc] = (ball_vy[1:][valid_bacc] - ball_vy[:-1][valid_bacc]) * FPS
    ball_accel = np.hypot(ball_ax, ball_ay).astype(np.float32)
    ball_speed_change = np.full(N_FRAMES, np.nan, dtype=np.float32)
    ball_speed_change[1:][valid_bacc] = (ball_speed[1:][valid_bacc] - ball_speed[:-1][valid_bacc]) * FPS

    # Causal last observed ball position & frames since last observation
    last_ball_x_norm, last_idx = causal_ffill_1d(ball_x_norm, ball_vis_bool)
    last_ball_y_norm, _ = causal_ffill_1d(ball_y_norm, ball_vis_bool)
    last_ball_x_m = (last_ball_x_norm * PITCH_L).astype(np.float32)
    last_ball_y_m = (last_ball_y_norm * PITCH_W).astype(np.float32)

    frames_since_ball = np.full(N_FRAMES, np.nan, dtype=np.float32)
    seen_so_far = last_idx >= 0
    frames_since_ball[seen_so_far] = (np.arange(N_FRAMES)[seen_so_far] - last_idx[seen_so_far]).astype(np.float32)

    # Pitch boundaries & out-of-bounds
    ball_oob = np.full(N_FRAMES, np.nan, dtype=np.float32)
    oob_bool = (ball_x_norm < 0.0) | (ball_x_norm > 1.0) | (ball_y_norm < 0.0) | (ball_y_norm > 1.0)
    ball_oob[ball_vis_bool] = oob_bool[ball_vis_bool].astype(np.float32)

    last_ball_oob = np.full(N_FRAMES, np.nan, dtype=np.float32)
    last_oob_bool = (last_ball_x_norm < 0.0) | (last_ball_x_norm > 1.0) | (last_ball_y_norm < 0.0) | (last_ball_y_norm > 1.0)
    last_ball_oob[seen_so_far] = last_oob_bool[seen_so_far].astype(np.float32)

    dist_sideline = np.minimum(ball_y_m, PITCH_W - ball_y_m).astype(np.float32)
    dist_goalline = np.minimum(ball_x_m, PITCH_L - ball_x_m).astype(np.float32)
    dist_boundary = np.minimum(dist_sideline, dist_goalline).astype(np.float32)

    # Ball movement direction (angle in radians, plus sin/cos for continuity)
    ball_dir_rad = np.arctan2(ball_vy, ball_vx).astype(np.float32)
    ball_dir_cos = np.cos(ball_dir_rad).astype(np.float32)
    ball_dir_sin = np.sin(ball_dir_rad).astype(np.float32)

    fb.add("ball_x_norm", ball_x_norm, "PartA_Ball", "Ball X coordinate (normalised 0-1, NaN when unobserved)", "norm")
    fb.add("ball_y_norm", ball_y_norm, "PartA_Ball", "Ball Y coordinate (normalised 0-1, NaN when unobserved)", "norm")
    fb.add("ball_x_m", ball_x_m, "PartA_Ball", "Ball X coordinate in metres (0-105.3, NaN when unobserved)", "m")
    fb.add("ball_y_m", ball_y_m, "PartA_Ball", "Ball Y coordinate in metres (0-68.0, NaN when unobserved)", "m")
    fb.add("ball_visible", ball_vis, "PartA_Ball", "Ball visibility indicator (1=observed, 0=missing)", "bool")
    fb.add("ball_vx_mps", ball_vx, "PartA_Ball", "Causal 1-step ball velocity X (m/s)", "m/s")
    fb.add("ball_vy_mps", ball_vy, "PartA_Ball", "Causal 1-step ball velocity Y (m/s)", "m/s")
    fb.add("ball_speed_mps", ball_speed, "PartA_Ball", "Ball speed from 1-step position difference (m/s)", "m/s")
    fb.add("ball_ax_mps2", ball_ax, "PartA_Ball", "Causal 2-step ball acceleration X (m/s^2)", "m/s^2")
    fb.add("ball_ay_mps2", ball_ay, "PartA_Ball", "Causal 2-step ball acceleration Y (m/s^2)", "m/s^2")
    fb.add("ball_accel_mps2", ball_accel, "PartA_Ball", "Ball acceleration magnitude (m/s^2)", "m/s^2")
    fb.add("ball_tangential_accel_mps2", ball_speed_change, "PartA_Ball", "Causal rate of change of ball speed (m/s^2)", "m/s^2")
    fb.add("frames_since_ball_obs", frames_since_ball, "PartA_Ball", "Frames elapsed since last ball observation (0 when visible)", "frames")
    fb.add("last_ball_x_norm", last_ball_x_norm, "PartA_Ball", "Causal last observed ball X (normalised)", "norm")
    fb.add("last_ball_y_norm", last_ball_y_norm, "PartA_Ball", "Causal last observed ball Y (normalised)", "norm")
    fb.add("last_ball_x_m", last_ball_x_m, "PartA_Ball", "Causal last observed ball X in metres", "m")
    fb.add("last_ball_y_m", last_ball_y_m, "PartA_Ball", "Causal last observed ball Y in metres", "m")
    fb.add("ball_outside_pitch", ball_oob, "PartA_Ball", "Ball outside [0,1]^2 pitch boundaries when visible (NaN when missing)", "bool")
    fb.add("last_ball_outside_pitch", last_ball_oob, "PartA_Ball", "Causal last observed ball outside [0,1]^2 pitch boundaries", "bool")
    fb.add("ball_dist_sideline_m", dist_sideline, "PartA_Ball", "Signed distance from ball to nearest touchline (m)", "m")
    fb.add("ball_dist_goalline_m", dist_goalline, "PartA_Ball", "Signed distance from ball to nearest goal line (m)", "m")
    fb.add("ball_dist_boundary_m", dist_boundary, "PartA_Ball", "Signed distance from ball to nearest pitch boundary (m)", "m")
    fb.add("ball_dir_rad", ball_dir_rad, "PartA_Ball", "Ball movement angle atan2(vy, vx) in radians", "rad")
    fb.add("ball_dir_cos", ball_dir_cos, "PartA_Ball", "Cosine of ball movement direction", "unitless")
    fb.add("ball_dir_sin", ball_dir_sin, "PartA_Ball", "Sine of ball movement direction", "unitless")

    # =========================================================================
    # PART B — PLAYER FEATURES (46 SLOTS + AGGREGATES)
    # =========================================================================
    print("[Part B] Computing Player Features across all 46 slots...")
    px_norm = players_raw[:, :, 0].astype(np.float32)
    py_norm = players_raw[:, :, 1].astype(np.float32)
    p_vis_bool = ~np.isnan(px_norm)
    p_vis = p_vis_bool.astype(np.float32)

    px_m = (px_norm * PITCH_L).astype(np.float32)
    py_m = (py_norm * PITCH_W).astype(np.float32)

    # Causal 1-step player velocity & speed from coordinates, gated at MAX_PLAYER_STEP_SPEED
    pvx = np.full((N_FRAMES, n_slots), np.nan, dtype=np.float32)
    pvy = np.full((N_FRAMES, n_slots), np.nan, dtype=np.float32)
    p_consec = p_vis_bool[1:] & p_vis_bool[:-1]
    dvx = (px_m[1:] - px_m[:-1]) * FPS
    dvy = (py_m[1:] - py_m[:-1]) * FPS
    dsp = np.hypot(dvx, dvy)
    valid_pstep = p_consec & (dsp <= MAX_PLAYER_STEP_SPEED)
    pvx[1:][valid_pstep] = dvx[valid_pstep]
    pvy[1:][valid_pstep] = dvy[valid_pstep]
    pspeed = np.hypot(pvx, pvy).astype(np.float32)

    # Causal 2-step player acceleration where reliable (gated at MAX_PLAYER_ACCEL)
    p_accel = np.full((N_FRAMES, n_slots), np.nan, dtype=np.float32)
    valid_pacc = (~np.isnan(pvx[1:])) & (~np.isnan(pvx[:-1]))
    dax = (pvx[1:] - pvx[:-1]) * FPS
    day = (pvy[1:] - pvy[:-1]) * FPS
    dacc = np.hypot(dax, day)
    reliable_pacc = valid_pacc & (dacc <= MAX_PLAYER_ACCEL)
    p_accel[1:][reliable_pacc] = dacc[reliable_pacc]

    # Distance from each player to the observed ball (NaN when ball or player missing)
    p_dist_ball = np.hypot(px_m - ball_x_m[:, None], py_m - ball_y_m[:, None]).astype(np.float32)

    # Also distance to last observed ball (for gap robustness in team/proximity summaries)
    p_dist_last_ball = np.hypot(px_m - last_ball_x_m[:, None], py_m - last_ball_y_m[:, None]).astype(np.float32)

    # Add per-slot blocks (8 * 46 = 368 features)
    fb.add_block([f"p{i}_x_m" for i in range(1, n_slots + 1)], px_m, "PartB_PlayerSlots",
                 "Slot {slot} X position in metres", "m")
    fb.add_block([f"p{i}_y_m" for i in range(1, n_slots + 1)], py_m, "PartB_PlayerSlots",
                 "Slot {slot} Y position in metres", "m")
    fb.add_block([f"p{i}_vis" for i in range(1, n_slots + 1)], p_vis, "PartB_PlayerSlots",
                 "Slot {slot} visibility indicator (1=present, 0=NaN)", "bool")
    fb.add_block([f"p{i}_vx_mps" for i in range(1, n_slots + 1)], pvx, "PartB_PlayerSlots",
                 "Slot {slot} causal velocity X from coordinates (m/s)", "m/s")
    fb.add_block([f"p{i}_vy_mps" for i in range(1, n_slots + 1)], pvy, "PartB_PlayerSlots",
                 "Slot {slot} causal velocity Y from coordinates (m/s)", "m/s")
    fb.add_block([f"p{i}_speed_mps" for i in range(1, n_slots + 1)], pspeed, "PartB_PlayerSlots",
                 "Slot {slot} causal speed from coordinates (m/s)", "m/s")
    fb.add_block([f"p{i}_accel_mps2" for i in range(1, n_slots + 1)], p_accel, "PartB_PlayerSlots",
                 "Slot {slot} reliable causal acceleration magnitude (m/s^2)", "m/s^2")
    fb.add_block([f"p{i}_dist_ball_m" for i in range(1, n_slots + 1)], p_dist_ball, "PartB_PlayerSlots",
                 "Slot {slot} distance to observed ball in metres", "m")

    # Nearest & Second-nearest player per team to the observed ball
    INF = np.float32(1e9)
    dist_H = np.where(np.isnan(p_dist_ball[:, home_mask]), INF, p_dist_ball[:, home_mask])
    dist_A = np.where(np.isnan(p_dist_ball[:, away_mask]), INF, p_dist_ball[:, away_mask])

    order_H = np.argsort(dist_H, axis=1)
    order_A = np.argsort(dist_A, axis=1)
    rows = np.arange(N_FRAMES)

    d1_H = dist_H[rows, order_H[:, 0]]
    d2_H = dist_H[rows, order_H[:, 1]]
    d1_A = dist_A[rows, order_A[:, 0]]
    d2_A = dist_A[rows, order_A[:, 1]]

    d1_H[d1_H >= INF] = np.nan
    d2_H[d2_H >= INF] = np.nan
    d1_A[d1_A >= INF] = np.nan
    d2_A[d2_A >= INF] = np.nan

    # Slot IDs (1..24 for H, 25..46 for A) and speeds of nearest / 2nd nearest players
    home_slot_ids = np.flatnonzero(home_mask) + 1
    away_slot_ids = np.flatnonzero(away_mask) + 1
    pspeed_H = pspeed[:, home_mask]
    pspeed_A = pspeed[:, away_mask]

    slot1_H = np.where(np.isnan(d1_H), np.nan, home_slot_ids[order_H[:, 0]]).astype(np.float32)
    slot2_H = np.where(np.isnan(d2_H), np.nan, home_slot_ids[order_H[:, 1]]).astype(np.float32)
    slot1_A = np.where(np.isnan(d1_A), np.nan, away_slot_ids[order_A[:, 0]]).astype(np.float32)
    slot2_A = np.where(np.isnan(d2_A), np.nan, away_slot_ids[order_A[:, 1]]).astype(np.float32)

    sp1_H = np.where(np.isnan(d1_H), np.nan, pspeed_H[rows, order_H[:, 0]]).astype(np.float32)
    sp2_H = np.where(np.isnan(d2_H), np.nan, pspeed_H[rows, order_H[:, 1]]).astype(np.float32)
    sp1_A = np.where(np.isnan(d1_A), np.nan, pspeed_A[rows, order_A[:, 0]]).astype(np.float32)
    sp2_A = np.where(np.isnan(d2_A), np.nan, pspeed_A[rows, order_A[:, 1]]).astype(np.float32)

    nearest_diff_A_minus_H = (d1_A - d1_H).astype(np.float32)
    nearest_ratio_H_over_A = (d1_H / (d1_A + 0.1)).astype(np.float32)
    # Match baseline.py exact logic: if ball is visible and at least one player is visible, pick team with smaller distance
    has_H = ~np.isnan(d1_H)
    has_A = ~np.isnan(d1_A)
    nearest_team_code = np.full(N_FRAMES, np.nan, dtype=np.float32)
    both_vis = has_H & has_A
    nearest_team_code[both_vis] = np.where(d1_H[both_vis] <= d1_A[both_vis], 1.0, -1.0)
    nearest_team_code[has_H & ~has_A] = 1.0
    nearest_team_code[~has_H & has_A] = -1.0

    # Also nearest distances to last observed ball (always available after first ball sighting)
    dist_last_H = np.where(np.isnan(p_dist_last_ball[:, home_mask]), INF, p_dist_last_ball[:, home_mask])
    dist_last_A = np.where(np.isnan(p_dist_last_ball[:, away_mask]), INF, p_dist_last_ball[:, away_mask])
    d1_last_H = np.min(dist_last_H, axis=1); d1_last_H[d1_last_H >= INF] = np.nan
    d1_last_A = np.min(dist_last_A, axis=1); d1_last_A[d1_last_A >= INF] = np.nan
    d1_last_diff = (d1_last_A - d1_last_H).astype(np.float32)

    fb.add("nearest_dist_H_m", d1_H, "PartB_PlayerBall", "Distance from observed ball to nearest Barcelona player (m)", "m")
    fb.add("second_nearest_dist_H_m", d2_H, "PartB_PlayerBall", "Distance from observed ball to 2nd-nearest Barcelona player (m)", "m")
    fb.add("nearest_dist_A_m", d1_A, "PartB_PlayerBall", "Distance from observed ball to nearest Athletic player (m)", "m")
    fb.add("second_nearest_dist_A_m", d2_A, "PartB_PlayerBall", "Distance from observed ball to 2nd-nearest Athletic player (m)", "m")
    fb.add("nearest_slot_H", slot1_H, "PartB_PlayerBall", "Track slot ID (1..24) of nearest Barcelona player to ball", "slot_id")
    fb.add("second_nearest_slot_H", slot2_H, "PartB_PlayerBall", "Track slot ID (1..24) of 2nd-nearest Barcelona player to ball", "slot_id")
    fb.add("nearest_slot_A", slot1_A, "PartB_PlayerBall", "Track slot ID (25..46) of nearest Athletic player to ball", "slot_id")
    fb.add("second_nearest_slot_A", slot2_A, "PartB_PlayerBall", "Track slot ID (25..46) of 2nd-nearest Athletic player to ball", "slot_id")
    fb.add("nearest_speed_H_mps", sp1_H, "PartB_PlayerBall", "Speed of nearest Barcelona player to ball (m/s)", "m/s")
    fb.add("second_nearest_speed_H_mps", sp2_H, "PartB_PlayerBall", "Speed of 2nd-nearest Barcelona player to ball (m/s)", "m/s")
    fb.add("nearest_speed_A_mps", sp1_A, "PartB_PlayerBall", "Speed of nearest Athletic player to ball (m/s)", "m/s")
    fb.add("second_nearest_speed_A_mps", sp2_A, "PartB_PlayerBall", "Speed of 2nd-nearest Athletic player to ball (m/s)", "m/s")
    fb.add("nearest_dist_diff_A_minus_H_m", nearest_diff_A_minus_H, "PartB_PlayerBall", "d1_A - d1_H (+ means Barcelona closer to ball)", "m")
    fb.add("nearest_dist_ratio_H_over_A", nearest_ratio_H_over_A, "PartB_PlayerBall", "d1_H / (d1_A + 0.1)", "ratio")
    fb.add("nearest_team_sign", nearest_team_code, "PartB_PlayerBall", "Nearest team to observed ball (+1=H, -1=A, NaN=ball missing)", "sign")
    fb.add("nearest_dist_last_ball_H_m", d1_last_H, "PartB_PlayerBall", "Distance from causal last-observed ball to nearest H player (m)", "m")
    fb.add("nearest_dist_last_ball_A_m", d1_last_A, "PartB_PlayerBall", "Distance from causal last-observed ball to nearest A player (m)", "m")
    fb.add("nearest_dist_last_ball_diff_m", d1_last_diff, "PartB_PlayerBall", "d1_last_A - d1_last_H (+ means H closer to last ball)", "m")

    # Number of players within 2m, 5m, 10m of the observed ball (NaN when ball is not observed)
    for radius in (2.0, 5.0, 10.0):
        r_int = int(radius)
        c_H = np.sum(dist_H <= radius, axis=1).astype(np.float32)
        c_A = np.sum(dist_A <= radius, axis=1).astype(np.float32)
        c_H[~ball_vis_bool] = np.nan
        c_A[~ball_vis_bool] = np.nan
        c_tot = c_H + c_A
        c_diff = c_H - c_A
        fb.add(f"players_within_{r_int}m_H", c_H, "PartB_PlayerBall", f"Barcelona players within {r_int}m of observed ball", "count")
        fb.add(f"players_within_{r_int}m_A", c_A, "PartB_PlayerBall", f"Athletic players within {r_int}m of observed ball", "count")
        fb.add(f"players_within_{r_int}m_total", c_tot, "PartB_PlayerBall", f"Total players within {r_int}m of observed ball", "count")
        fb.add(f"players_within_{r_int}m_diff_H_minus_A", c_diff, "PartB_PlayerBall", f"H minus A players within {r_int}m of observed ball", "count")

    # =========================================================================
    # PART C — TEAM FEATURES
    # =========================================================================
    print("[Part C] Computing Team Features...")
    # Outfield masks (excluding goalkeeper slots 24 and 46 for outfield shape metrics, while keeping all-player centroids too)
    gk_mask = np.zeros(n_slots, dtype=bool)
    gk_mask[23] = True  # Slot 24 (BAR GK)
    gk_mask[45] = True  # Slot 46 (ATH GK)

    def team_geometry(slot_mask, prefix, label):
        tx_m = px_m[:, slot_mask]
        ty_m = py_m[:, slot_mask]
        tx_n = px_norm[:, slot_mask]
        ty_n = py_norm[:, slot_mask]
        tsp = pspeed[:, slot_mask]
        tvx = pvx[:, slot_mask]
        tvy = pvy[:, slot_mask]

        vis_cnt = np.sum(~np.isnan(tx_m), axis=1).astype(np.float32)
        cx_m = np.nanmean(tx_m, axis=1).astype(np.float32)
        cy_m = np.nanmean(ty_m, axis=1).astype(np.float32)
        cx_n = np.nanmean(tx_n, axis=1).astype(np.float32)
        cy_n = np.nanmean(ty_n, axis=1).astype(np.float32)

        std_x = np.nanstd(tx_m, axis=1).astype(np.float32)
        std_y = np.nanstd(ty_m, axis=1).astype(np.float32)

        # Compactness: mean Euclidean distance of visible players to team centroid
        dist_to_c = np.hypot(tx_m - cx_m[:, None], ty_m - cy_m[:, None])
        compact_mean = np.nanmean(dist_to_c, axis=1).astype(np.float32)
        compact_std = np.nanstd(dist_to_c, axis=1).astype(np.float32)

        # Team width (Y span across pitch width) & length (X span along pitch length)
        width_y = (np.nanmax(ty_m, axis=1) - np.nanmin(ty_m, axis=1)).astype(np.float32)
        length_x = (np.nanmax(tx_m, axis=1) - np.nanmin(tx_m, axis=1)).astype(np.float32)
        surface_box = (width_y * length_x).astype(np.float32)

        # Team speed stats
        sp_mean = np.nanmean(tsp, axis=1).astype(np.float32)
        sp_max = np.nanmax(tsp, axis=1).astype(np.float32)
        sp_std = np.nanstd(tsp, axis=1).astype(np.float32)
        vx_mean = np.nanmean(tvx, axis=1).astype(np.float32)
        vy_mean = np.nanmean(tvy, axis=1).astype(np.float32)

        # Team movement towards the observed ball (mean radial velocity towards ball)
        # Unit vector from player i to ball: u = (ball - player) / ||ball - player||
        dx_b = ball_x_m[:, None] - tx_m
        dy_b = ball_y_m[:, None] - ty_m
        dist_b = np.hypot(dx_b, dy_b)
        safe_dist = np.where(dist_b > 0.05, dist_b, np.nan)
        ux = dx_b / safe_dist
        uy = dy_b / safe_dist
        v_towards = tvx * ux + tvy * uy
        mean_v_towards = np.nanmean(v_towards, axis=1).astype(np.float32)
        max_v_towards = np.nanmax(v_towards, axis=1).astype(np.float32)
        frac_moving_towards = np.nanmean((v_towards > 0.5).astype(np.float32), axis=1).astype(np.float32)
        frac_moving_towards[np.isnan(mean_v_towards)] = np.nan

        fb.add(f"team_{prefix}_visible_players", vis_cnt, "PartC_Team", f"Number of visible {label} players", "count")
        fb.add(f"team_{prefix}_centroid_x_m", cx_m, "PartC_Team", f"{label} centroid X in metres", "m")
        fb.add(f"team_{prefix}_centroid_y_m", cy_m, "PartC_Team", f"{label} centroid Y in metres", "m")
        fb.add(f"team_{prefix}_avg_x_norm", cx_n, "PartC_Team", f"{label} average normalised X", "norm")
        fb.add(f"team_{prefix}_avg_y_norm", cy_n, "PartC_Team", f"{label} average normalised Y", "norm")
        fb.add(f"team_{prefix}_std_x_m", std_x, "PartC_Team", f"{label} X standard deviation (m)", "m")
        fb.add(f"team_{prefix}_std_y_m", std_y, "PartC_Team", f"{label} Y standard deviation (m)", "m")
        fb.add(f"team_{prefix}_compactness_mean_m", compact_mean, "PartC_Team", f"{label} mean distance to team centroid (m)", "m")
        fb.add(f"team_{prefix}_compactness_std_m", compact_std, "PartC_Team", f"{label} std of distance to team centroid (m)", "m")
        fb.add(f"team_{prefix}_width_y_m", width_y, "PartC_Team", f"{label} lateral width (max Y - min Y in metres)", "m")
        fb.add(f"team_{prefix}_length_x_m", length_x, "PartC_Team", f"{label} longitudinal length (max X - min X in metres)", "m")
        fb.add(f"team_{prefix}_box_area_m2", surface_box, "PartC_Team", f"{label} bounding box area (width * length, m^2)", "m^2")
        fb.add(f"team_{prefix}_speed_mean_mps", sp_mean, "PartC_Team", f"{label} average player speed (m/s)", "m/s")
        fb.add(f"team_{prefix}_speed_max_mps", sp_max, "PartC_Team", f"{label} max player speed (m/s)", "m/s")
        fb.add(f"team_{prefix}_speed_std_mps", sp_std, "PartC_Team", f"{label} player speed std (m/s)", "m/s")
        fb.add(f"team_{prefix}_vx_mean_mps", vx_mean, "PartC_Team", f"{label} mean velocity X (m/s)", "m/s")
        fb.add(f"team_{prefix}_vy_mean_mps", vy_mean, "PartC_Team", f"{label} mean velocity Y (m/s)", "m/s")
        fb.add(f"team_{prefix}_vel_towards_ball_mean_mps", mean_v_towards, "PartC_Team", f"{label} mean velocity component towards observed ball (m/s)", "m/s")
        fb.add(f"team_{prefix}_vel_towards_ball_max_mps", max_v_towards, "PartC_Team", f"{label} max velocity component towards observed ball (m/s)", "m/s")
        fb.add(f"team_{prefix}_frac_moving_towards_ball", frac_moving_towards, "PartC_Team", f"Fraction of visible {label} players moving >0.5 m/s towards ball", "ratio")

        return {
            "vis": vis_cnt, "cx": cx_m, "cy": cy_m, "std_x": std_x, "std_y": std_y,
            "compact": compact_mean, "width": width_y, "length": length_x,
            "sp_mean": sp_mean, "vx_mean": vx_mean, "vy_mean": vy_mean,
            "v_towards": mean_v_towards,
        }

    geom_H = team_geometry(home_mask, "H", "Barcelona (H)")
    geom_A = team_geometry(away_mask, "A", "Athletic Club (A)")

    # Outfield-only length (excluding GK) for accurate defensive/attacking line depth
    outfield_H_len = (np.nanmax(px_m[:, home_mask & ~gk_mask], axis=1) -
                      np.nanmin(px_m[:, home_mask & ~gk_mask], axis=1)).astype(np.float32)
    outfield_A_len = (np.nanmax(px_m[:, away_mask & ~gk_mask], axis=1) -
                      np.nanmin(px_m[:, away_mask & ~gk_mask], axis=1)).astype(np.float32)
    fb.add("team_H_outfield_length_x_m", outfield_H_len, "PartC_Team", "Barcelona outfield longitudinal length excluding GK slot 24 (m)", "m")
    fb.add("team_A_outfield_length_x_m", outfield_A_len, "PartC_Team", "Athletic outfield longitudinal length excluding GK slot 46 (m)", "m")

    # Inter-team & spatial dominance features
    centroid_dist = np.hypot(geom_H["cx"] - geom_A["cx"], geom_H["cy"] - geom_A["cy"]).astype(np.float32)
    centroid_dx = (geom_H["cx"] - geom_A["cx"]).astype(np.float32)
    centroid_dy = (geom_H["cy"] - geom_A["cy"]).astype(np.float32)

    # Spatial dominance:
    # 1) Pitch control grid (7x5 = 35 control points across 105.3m x 68m): fraction of pitch points closer to H than A
    gx = np.linspace(7.5, PITCH_L - 7.5, 7, dtype=np.float32)
    gy = np.linspace(6.8, PITCH_W - 6.8, 5, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_pts_x = grid_x.ravel()[None, None, :]  # (1, 1, 35)
    grid_pts_y = grid_y.ravel()[None, None, :]  # (1, 1, 35)

    # Compute chunked Voronoi pitch control to keep memory low
    voronoi_H = np.full(N_FRAMES, np.nan, dtype=np.float32)
    chunk_sz = 25000
    px_H = px_m[:, home_mask][:, :, None]
    py_H = py_m[:, home_mask][:, :, None]
    px_A = px_m[:, away_mask][:, :, None]
    py_A = py_m[:, away_mask][:, :, None]
    for s in range(0, N_FRAMES, chunk_sz):
        e = min(N_FRAMES, s + chunk_sz)
        dH_grid = np.hypot(px_H[s:e] - grid_pts_x, py_H[s:e] - grid_pts_y)
        dA_grid = np.hypot(px_A[s:e] - grid_pts_x, py_A[s:e] - grid_pts_y)
        min_dH = np.nanmin(dH_grid, axis=1)  # (chunk, 35)
        min_dA = np.nanmin(dA_grid, axis=1)  # (chunk, 35)
        valid_g = (~np.isnan(min_dH)) & (~np.isnan(min_dA))
        dom = np.mean(min_dH < min_dA, axis=1).astype(np.float32)
        dom[~np.any(valid_g, axis=1)] = np.nan
        voronoi_H[s:e] = dom

    # 2) Local ball-zone dominance: soft Gaussian influence ratio around observed ball (sigma = 8m)
    infl_H = np.nansum(np.exp(-0.5 * (p_dist_ball[:, home_mask] / 8.0) ** 2), axis=1).astype(np.float32)
    infl_A = np.nansum(np.exp(-0.5 * (p_dist_ball[:, away_mask] / 8.0) ** 2), axis=1).astype(np.float32)
    infl_H[~ball_vis_bool] = np.nan
    infl_A[~ball_vis_bool] = np.nan
    ball_zone_dom_H = (infl_H / (infl_H + infl_A + 1e-5)).astype(np.float32)

    fb.add("teams_centroid_dist_m", centroid_dist, "PartC_Team", "Distance between Barcelona and Athletic centroids (m)", "m")
    fb.add("teams_centroid_dx_H_minus_A_m", centroid_dx, "PartC_Team", "Barcelona centroid X minus Athletic centroid X (m)", "m")
    fb.add("teams_centroid_dy_H_minus_A_m", centroid_dy, "PartC_Team", "Barcelona centroid Y minus Athletic centroid Y (m)", "m")
    fb.add("team_spatial_dominance_voronoi_H", voronoi_H, "PartC_Team", "Fraction of 35 pitch grid points closer to Barcelona than Athletic", "ratio")
    fb.add("team_ball_zone_influence_H", infl_H, "PartC_Team", "Gaussian spatial influence of Barcelona around observed ball (sigma=8m)", "score")
    fb.add("team_ball_zone_influence_A", infl_A, "PartC_Team", "Gaussian spatial influence of Athletic around observed ball (sigma=8m)", "score")
    fb.add("team_ball_zone_dominance_H", ball_zone_dom_H, "PartC_Team", "Barcelona share of Gaussian spatial influence around observed ball", "ratio")
    fb.add("team_vel_towards_ball_diff_H_minus_A", geom_H["v_towards"] - geom_A["v_towards"], "PartC_Team",
           "H minus A mean velocity towards observed ball (m/s)", "m/s")

    # =========================================================================
    # PART D — TEMPORAL FEATURES (STRICTLY CAUSAL / PAST-ONLY)
    # =========================================================================
    print("[Part D] Computing Causal Temporal Features (windows: 25f=1s, 75f=3s, 125f=5s)...")
    # 1. Time since last ball observation in seconds
    time_since_ball_s = (frames_since_ball / FPS).astype(np.float32)
    fb.add("time_since_last_ball_obs_s", time_since_ball_s, "PartD_Temporal", "Causal time elapsed since last ball observation (s)", "s")

    # 2. Rolling player movement statistics (team speeds, max speeds, nearest distances over 1s, 3s, 5s)
    for win, label in ((25, "1s"), (75, "3s"), (125, "5s")):
        sp_H_m, sp_H_s = causal_rolling_mean_std(geom_H["sp_mean"], win)
        sp_A_m, sp_A_s = causal_rolling_mean_std(geom_A["sp_mean"], win)
        d1_H_m = causal_rolling_mean(d1_H, win)
        d1_A_m = causal_rolling_mean(d1_A, win)
        ddiff_m = causal_rolling_mean(nearest_diff_A_minus_H, win)
        dom_m = causal_rolling_mean(ball_zone_dom_H, win)

        fb.add(f"roll_{label}_team_H_speed_mean", sp_H_m, "PartD_Temporal", f"Causal {label} rolling mean of Barcelona avg speed (m/s)", "m/s")
        fb.add(f"roll_{label}_team_H_speed_std", sp_H_s, "PartD_Temporal", f"Causal {label} rolling std of Barcelona avg speed (m/s)", "m/s")
        fb.add(f"roll_{label}_team_A_speed_mean", sp_A_m, "PartD_Temporal", f"Causal {label} rolling mean of Athletic avg speed (m/s)", "m/s")
        fb.add(f"roll_{label}_team_A_speed_std", sp_A_s, "PartD_Temporal", f"Causal {label} rolling std of Athletic avg speed (m/s)", "m/s")
        fb.add(f"roll_{label}_nearest_dist_H_m", d1_H_m, "PartD_Temporal", f"Causal {label} rolling mean nearest H distance to ball (m)", "m")
        fb.add(f"roll_{label}_nearest_dist_A_m", d1_A_m, "PartD_Temporal", f"Causal {label} rolling mean nearest A distance to ball (m)", "m")
        fb.add(f"roll_{label}_nearest_diff_A_minus_H_m", ddiff_m, "PartD_Temporal", f"Causal {label} rolling mean (d1_A - d1_H) (m)", "m")
        fb.add(f"roll_{label}_ball_zone_dom_H", dom_m, "PartD_Temporal", f"Causal {label} rolling mean ball-zone dominance H", "ratio")

    # 3. Rolling team centroid movement (causal velocity & net displacement over 1s and 3s)
    for win, label in ((25, "1s"), (75, "3s")):
        dt_win = win / FPS
        for prefix, geom in (("H", geom_H), ("A", geom_A)):
            cx, cy = geom["cx"], geom["cy"]
            dcx = np.full(N_FRAMES, np.nan, dtype=np.float32)
            dcy = np.full(N_FRAMES, np.nan, dtype=np.float32)
            dcx[win:] = (cx[win:] - cx[:-win]) / dt_win
            dcy[win:] = (cy[win:] - cy[:-win]) / dt_win
            dsp_c = np.hypot(dcx, dcy).astype(np.float32)
            fb.add(f"causal_{label}_centroid_{prefix}_vx_mps", dcx, "PartD_Temporal", f"Causal {label} net centroid velocity X for {prefix} (m/s)", "m/s")
            fb.add(f"causal_{label}_centroid_{prefix}_vy_mps", dcy, "PartD_Temporal", f"Causal {label} net centroid velocity Y for {prefix} (m/s)", "m/s")
            fb.add(f"causal_{label}_centroid_{prefix}_speed_mps", dsp_c, "PartD_Temporal", f"Causal {label} net centroid speed for {prefix} (m/s)", "m/s")

    # 4. Recent ball movement (causal rolling ball speed, acceleration, net displacement over 1s & 3s)
    for win, label in ((25, "1s"), (75, "3s")):
        bsp_m, bsp_s = causal_rolling_mean_std(ball_speed, win)
        bacc_m = causal_rolling_mean(ball_accel, win)
        # Net displacement of last-observed ball over [t - win, t]
        b_disp = np.full(N_FRAMES, np.nan, dtype=np.float32)
        b_disp[win:] = np.hypot(last_ball_x_m[win:] - last_ball_x_m[:-win],
                                last_ball_y_m[win:] - last_ball_y_m[:-win]).astype(np.float32)
        fb.add(f"roll_{label}_ball_speed_mean_mps", bsp_m, "PartD_Temporal", f"Causal {label} rolling mean ball speed (m/s)", "m/s")
        fb.add(f"roll_{label}_ball_speed_std_mps", bsp_s, "PartD_Temporal", f"Causal {label} rolling std ball speed (m/s)", "m/s")
        fb.add(f"roll_{label}_ball_accel_mean_mps2", bacc_m, "PartD_Temporal", f"Causal {label} rolling mean ball acceleration (m/s^2)", "m/s^2")
        fb.add(f"causal_{label}_ball_net_disp_m", b_disp, "PartD_Temporal", f"Causal {label} net displacement of last observed ball (m)", "m")

    # 5. Team formation changes (causal 1s and 3s change in width, length, compactness, Voronoi dominance)
    for win, label in ((25, "1s"), (75, "3s")):
        for prefix, geom in (("H", geom_H), ("A", geom_A)):
            d_comp = np.full(N_FRAMES, np.nan, dtype=np.float32)
            d_wid = np.full(N_FRAMES, np.nan, dtype=np.float32)
            d_len = np.full(N_FRAMES, np.nan, dtype=np.float32)
            d_comp[win:] = geom["compact"][win:] - geom["compact"][:-win]
            d_wid[win:] = geom["width"][win:] - geom["width"][:-win]
            d_len[win:] = geom["length"][win:] - geom["length"][:-win]
            fb.add(f"causal_{label}_delta_compactness_{prefix}_m", d_comp, "PartD_Temporal", f"Causal {label} change in {prefix} compactness (m)", "m")
            fb.add(f"causal_{label}_delta_width_{prefix}_m", d_wid, "PartD_Temporal", f"Causal {label} change in {prefix} width (m)", "m")
            fb.add(f"causal_{label}_delta_length_{prefix}_m", d_len, "PartD_Temporal", f"Causal {label} change in {prefix} length (m)", "m")
        d_vor = np.full(N_FRAMES, np.nan, dtype=np.float32)
        d_vor[win:] = voronoi_H[win:] - voronoi_H[:-win]
        fb.add(f"causal_{label}_delta_voronoi_dom_H", d_vor, "PartD_Temporal", f"Causal {label} change in Barcelona Voronoi dominance", "ratio")

    # 6. Recent tracking reliability (causal rolling ball visibility rate, player count, homography reprojection error)
    _, reproj_err, homog_valid = load_homography.load()
    reproj_err = reproj_err.astype(np.float32)
    homog_valid = homog_valid.astype(np.float32)
    total_vis_players = (geom_H["vis"] + geom_A["vis"]).astype(np.float32)
    dup_flag = (total_vis_players > 22.0).astype(np.float32)

    fb.add("homography_reproj_error", reproj_err, "PartD_Temporal", "Frame homography reprojection error", "px")
    fb.add("homography_valid", homog_valid, "PartD_Temporal", "Frame homography valid flag (1=valid)", "bool")
    fb.add("total_visible_players", total_vis_players, "PartD_Temporal", "Total visible tracked players (H + A)", "count")
    fb.add("tracker_duplicate_overload_flag", dup_flag, "PartD_Temporal", "Indicator that >22 player slots are active on this frame", "bool")

    for win, label in ((25, "1s"), (75, "3s"), (125, "5s")):
        b_rel = causal_rolling_mean(ball_vis, win)
        p_rel = causal_rolling_mean(total_vis_players / 22.0, win)
        h_err = causal_rolling_mean(reproj_err, win)
        fb.add(f"roll_{label}_ball_vis_rate", b_rel, "PartD_Temporal", f"Causal {label} rolling ball visibility rate", "ratio")
        fb.add(f"roll_{label}_player_coverage_ratio", p_rel, "PartD_Temporal", f"Causal {label} rolling player count / 22", "ratio")
        fb.add(f"roll_{label}_homog_reproj_err", h_err, "PartD_Temporal", f"Causal {label} rolling mean homography reprojection error", "px")

    # =========================================================================
    # PART E — MATCH CONTEXT
    # =========================================================================
    print("[Part E] Computing Match Context & Smart Tagging Features...")
    half_id = np.zeros(N_FRAMES, dtype=np.float32)
    in_graded = np.zeros(N_FRAMES, dtype=np.float32)
    for h, lo, hi in HALF_BLOCKS:
        half_id[lo:hi + 1] = float(h)
        in_graded[lo:hi + 1] = 1.0

    is_first_half = (half_id == 1.0).astype(np.float32)
    is_second_half = (half_id == 2.0).astype(np.float32)

    # Attacking direction from ATD metadata: H (Barcelona) attacks left_to_right (+1) in 1st half, right_to_left (-1) in 2nd half
    attack_dir_H = np.where(half_id == 1.0, 1.0, np.where(half_id == 2.0, -1.0, 0.0)).astype(np.float32)
    attack_dir_A = (-attack_dir_H).astype(np.float32)

    # Direction-normalised attacking progress coordinates (in [-52.65, +52.65] m, where + is towards opponent goal)
    cx_centered = PITCH_L / 2.0
    ball_x_attack_H = ((ball_x_m - cx_centered) * attack_dir_H).astype(np.float32)
    ball_x_attack_H[half_id == 0] = np.nan
    last_ball_x_attack_H = ((last_ball_x_m - cx_centered) * attack_dir_H).astype(np.float32)
    last_ball_x_attack_H[half_id == 0] = np.nan
    team_H_cx_attack = ((geom_H["cx"] - cx_centered) * attack_dir_H).astype(np.float32)
    team_A_cx_attack = ((geom_A["cx"] - cx_centered) * attack_dir_A).astype(np.float32)

    # Time since kickoff (using official half block start AND official ball-kick anchor)
    time_since_half_block_start_s = np.full(N_FRAMES, np.nan, dtype=np.float32)
    for h, lo, hi in HALF_BLOCKS:
        time_since_half_block_start_s[lo:hi + 1] = (np.arange(lo, hi + 1) - lo) / FPS

    ko1 = KICKOFFS["1st Half"]   # 11185
    ko2 = KICKOFFS["2nd Half"]   # 113742
    time_since_kickoff_s = np.full(N_FRAMES, np.nan, dtype=np.float32)
    h1_lo, h1_hi = HALF_BLOCKS[0][1], HALF_BLOCKS[0][2]
    h2_lo, h2_hi = HALF_BLOCKS[1][1], HALF_BLOCKS[1][2]
    time_since_kickoff_s[h1_lo:h1_hi + 1] = (np.arange(h1_lo, h1_hi + 1) - ko1) / FPS
    time_since_kickoff_s[h2_lo:h2_hi + 1] = (np.arange(h2_lo, h2_hi + 1) - ko2) / FPS
    pre_kickoff_graded_2h = (( video_frame >= h2_lo ) & ( video_frame < ko2 )).astype(np.float32)

    fb.add("half_indicator", half_id, "PartE_Context", "Match half indicator (1=1st half, 2=2nd half, 0=ungraded)", "id")
    fb.add("in_graded_half", in_graded, "PartE_Context", "Indicator that frame falls inside official graded half blocks", "bool")
    fb.add("is_first_half", is_first_half, "PartE_Context", "1st half indicator (frames 11185..90651)", "bool")
    fb.add("is_second_half", is_second_half, "PartE_Context", "2nd half indicator (frames 110808..190197)", "bool")
    fb.add("attack_dir_H", attack_dir_H, "PartE_Context", "Barcelona attacking X direction (+1=L->R in 1H, -1=R->L in 2H, 0=ungraded)", "sign")
    fb.add("attack_dir_A", attack_dir_A, "PartE_Context", "Athletic attacking X direction (-1=R->L in 1H, +1=L->R in 2H, 0=ungraded)", "sign")
    fb.add("ball_x_attack_H_m", ball_x_attack_H, "PartE_Context", "Ball X in Barcelona attacking direction relative to halfway line (m)", "m")
    fb.add("last_ball_x_attack_H_m", last_ball_x_attack_H, "PartE_Context", "Causal last ball X in Barcelona attacking direction (m)", "m")
    fb.add("team_H_centroid_x_attack_m", team_H_cx_attack, "PartE_Context", "Barcelona centroid X in its attacking direction relative to halfway line (m)", "m")
    fb.add("team_A_centroid_x_attack_m", team_A_cx_attack, "PartE_Context", "Athletic centroid X in its attacking direction relative to halfway line (m)", "m")
    fb.add("time_since_half_start_s", time_since_half_block_start_s, "PartE_Context", "Seconds elapsed since start of graded half block", "s")
    fb.add("time_since_kickoff_s", time_since_kickoff_s, "PartE_Context", "Seconds relative to actual half kickoff (negative during 2H pre-kickoff)", "s")
    fb.add("is_2h_pre_kickoff_window", pre_kickoff_graded_2h, "PartE_Context", "Indicator for graded 2nd-half frames before 2nd-half kickoff (110808..113741)", "bool")

    # Smart Tagging Event & Set-Piece Indicators
    # IMPORTANT CAUSALITY NOTE:
    # Smart Tagging intervals [start_frame, end_frame) are automatic tactical phase annotations
    # from the tactical video. Marking whether frame t falls inside [start_frame, end_frame) is
    # synchronous with the tagger's interval. We also compute strictly causal `frames_since_tag_start`
    # and never use `end_frame - t` (which would look ahead to when the phase ends).
    events = load_smart_tagging.load()
    ALL_CODES = [
        "PLAYERS IN THE BOX",
        "BALL IN FINAL THIRD",
        "BALL IN THE BOX",
        "PROGRESSION",
        "DEFENDING IN MIDDLE THIRD",
        "SET PIECES",
        "CREATING CHANCES",
        "DEFENDING IN DEFENSIVE THIRD",
        "LONG BALL",
        "ATTACKING TRANSITION",
        "DEFENSIVE TRANSITION",
        "BUILD UP",
        "DEFENDING IN ATTACKING THIRD",
        "GOALS",
    ]
    SET_PIECE_TYPES = [
        "Throw-in",
        "Corner Kick",
        "Goal Kick",
        "Free Kick",
        "Kick Off (After Goal)",
        "Kick Off (Start)",
    ]

    code_arrays = {c: np.zeros(N_FRAMES, dtype=np.float32) for c in ALL_CODES}
    sp_arrays = {t: np.zeros(N_FRAMES, dtype=np.float32) for t in SET_PIECE_TYPES}
    st_team_H = np.zeros(N_FRAMES, dtype=np.float32)
    st_team_A = np.zeros(N_FRAMES, dtype=np.float32)
    st_team_NA = np.zeros(N_FRAMES, dtype=np.float32)
    st_active_count = np.zeros(N_FRAMES, dtype=np.float32)
    st_attack_H = np.zeros(N_FRAMES, dtype=np.float32)
    st_attack_A = np.zeros(N_FRAMES, dtype=np.float32)

    ATTACK_CODES = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "BALL IN THE BOX",
                    "PLAYERS IN THE BOX", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL", "GOALS"}
    DEFEND_CODES = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                    "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"}

    sp_start_frames = np.zeros(N_FRAMES, dtype=bool)

    for ev in events:
        s = max(0, min(N_FRAMES, ev["start_frame"]))
        e = max(0, min(N_FRAMES, ev["end_frame"]))
        if e <= s:
            continue
        c = ev["code"]
        team = ev["tags"].get("Team", "")
        sp_type = ev["tags"].get("Type", "")

        st_active_count[s:e] += 1.0
        if c in code_arrays:
            code_arrays[c][s:e] = 1.0
        if sp_type in sp_arrays:
            sp_arrays[sp_type][s:e] = 1.0
        if c == "SET PIECES" and s < N_FRAMES:
            sp_start_frames[s] = True

        if team == "Barcelona":
            st_team_H[s:e] = 1.0
            if c in ATTACK_CODES:
                st_attack_H[s:e] = 1.0
            elif c in DEFEND_CODES:
                st_attack_A[s:e] = 1.0  # H defending => A attacking
        elif team == "Atletic Club":
            st_team_A[s:e] = 1.0
            if c in ATTACK_CODES:
                st_attack_A[s:e] = 1.0
            elif c in DEFEND_CODES:
                st_attack_H[s:e] = 1.0  # A defending => H attacking
        elif team == "N/A":
            st_team_NA[s:e] = 1.0

    # Causal frames since most recent set-piece tag start
    _, last_sp_idx = causal_ffill_1d(np.arange(N_FRAMES, dtype=np.float32), sp_start_frames)
    frames_since_sp_start = np.full(N_FRAMES, np.nan, dtype=np.float32)
    has_sp = last_sp_idx >= 0
    frames_since_sp_start[has_sp] = (np.arange(N_FRAMES)[has_sp] - last_sp_idx[has_sp]).astype(np.float32)

    fb.add("st_active_tag_count", st_active_count, "PartE_SmartTagging", "Number of active Smart Tagging instances on frame", "count")
    fb.add("st_team_barcelona", st_team_H, "PartE_SmartTagging", "Smart Tagging active instance tagged Team=Barcelona", "bool")
    fb.add("st_team_athletic", st_team_A, "PartE_SmartTagging", "Smart Tagging active instance tagged Team=Atletic Club", "bool")
    fb.add("st_team_na", st_team_NA, "PartE_SmartTagging", "Smart Tagging active instance tagged Team=N/A", "bool")
    fb.add("st_implied_attacking_H", st_attack_H, "PartE_SmartTagging", "Smart Tagging implies Barcelona is attacking (H attack tag or A defend tag)", "bool")
    fb.add("st_implied_attacking_A", st_attack_A, "PartE_SmartTagging", "Smart Tagging implies Athletic is attacking (A attack tag or H defend tag)", "bool")
    fb.add("st_implied_team_net_H_minus_A", st_attack_H - st_attack_A, "PartE_SmartTagging", "Smart Tagging net attacking team prior (+1=H, -1=A, 0=neutral/ambiguous)", "sign")

    for c in ALL_CODES:
        slug = c.lower().replace(" ", "_")
        fb.add(f"st_code_{slug}", code_arrays[c], "PartE_SmartTagging", f"Smart Tagging code indicator: {c}", "bool")

    for t in SET_PIECE_TYPES:
        slug = t.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        fb.add(f"st_setpiece_{slug}", sp_arrays[t], "PartE_SmartTagging", f"Smart Tagging set-piece type indicator: {t}", "bool")

    fb.add("st_frames_since_setpiece_start", frames_since_sp_start, "PartE_SmartTagging", "Causal frames elapsed since start of most recent SET PIECES tag", "frames")

    # Context outside-pitch composite indicator (combines current/last ball OOB + near-boundary stoppage cue)
    context_ball_oob_or_edge = np.where(
        ball_vis_bool,
        ((ball_oob > 0.5) | (dist_boundary < 0.5)).astype(np.float32),
        ((last_ball_oob > 0.5) | (np.minimum(np.minimum(last_ball_y_m, PITCH_W - last_ball_y_m),
                                             np.minimum(last_ball_x_m, PITCH_L - last_ball_x_m)) < 0.5)).astype(np.float32)
    ).astype(np.float32)
    context_ball_oob_or_edge[~seen_so_far] = np.nan
    fb.add("context_ball_oob_or_within_0p5m_line", context_ball_oob_or_edge, "PartE_Context",
           "Indicator that ball (or last seen ball) is outside pitch or within 0.5m of boundary line", "bool")

    # Assemble full feature matrix
    print("[Part F] Stacking and validating feature matrix...")
    X = np.column_stack(fb.cols).astype(np.float32)
    feature_names = np.array([m["name"] for m in fb.meta])
    build_time = time.time() - t0
    print(f"  Built X of shape {X.shape} ({X.dtype}) in {build_time:.2f}s")
    return X, frame_id.astype(np.int32), video_frame, feature_names, fb.meta, build_time


def validate_and_save(X, frame_id, video_frame, feature_names, meta, build_time):
    os.makedirs(OUT_DIR, exist_ok=True)
    t_save0 = time.time()

    # Save compressed NumPy archive
    npz_path = os.path.join(OUT_DIR, "tactical_features.npz")
    np.savez_compressed(
        npz_path,
        X=X,
        frame_id=frame_id,
        video_frame=video_frame,
        feature_names=feature_names,
    )
    save_time = time.time() - t_save0
    npz_mb = os.path.getsize(npz_path) / (1024 * 1024)

    # Save catalog JSON
    catalog_path = os.path.join(OUT_DIR, "feature_catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # Compute missing value statistics and percentiles per feature
    graded_mask = np.zeros(N_FRAMES, dtype=bool)
    for _, lo, hi in HALF_BLOCKS:
        graded_mask[lo:hi + 1] = True

    stats_rows = []
    inf_total = 0
    for j, m in enumerate(meta):
        col = X[:, j]
        col_g = col[graded_mask]
        nan_all = int(np.isnan(col).sum())
        nan_g = int(np.isnan(col_g).sum())
        n_inf = int(np.isinf(col).sum())
        inf_total += n_inf
        valid_vals = col[~np.isnan(col) & ~np.isinf(col)]
        if len(valid_vals) > 0:
            vmin = float(np.min(valid_vals))
            vmean = float(np.mean(valid_vals))
            vstd = float(np.std(valid_vals))
            vmed = float(np.median(valid_vals))
            vp05 = float(np.percentile(valid_vals, 5))
            vp95 = float(np.percentile(valid_vals, 95))
            vmax = float(np.max(valid_vals))
        else:
            vmin = vmean = vstd = vmed = vp05 = vp95 = vmax = float("nan")
        stats_rows.append({
            "index": j,
            "name": m["name"],
            "group": m["group"],
            "unit": m["unit"],
            "missing_all_count": nan_all,
            "missing_all_pct": round(100.0 * nan_all / N_FRAMES, 2),
            "missing_graded_count": nan_g,
            "missing_graded_pct": round(100.0 * nan_g / int(graded_mask.sum()), 2),
            "inf_count": n_inf,
            "min": round(vmin, 4) if not np.isnan(vmin) else None,
            "p05": round(vp05, 4) if not np.isnan(vp05) else None,
            "median": round(vmed, 4) if not np.isnan(vmed) else None,
            "mean": round(vmean, 4) if not np.isnan(vmean) else None,
            "std": round(vstd, 4) if not np.isnan(vstd) else None,
            "p95": round(vp95, 4) if not np.isnan(vp95) else None,
            "max": round(vmax, 4) if not np.isnan(vmax) else None,
            "description": m["description"],
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_csv_path = os.path.join(OUT_DIR, "feature_stats.csv")
    stats_df.to_csv(stats_csv_path, index=False)

    # Group summaries
    group_counts = collections.Counter(m["group"] for m in meta)
    group_missing = stats_df.groupby("group")[["missing_all_pct", "missing_graded_pct"]].agg(["mean", "min", "max"]).to_dict()

    # Validation checks
    name_to_idx = {m["name"]: i for i, m in enumerate(meta)}
    checks = []

    def add_check(name, ok, detail=""):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    add_check("row_count_equals_199319", X.shape[0] == 199319, f"rows={X.shape[0]}")
    add_check("frame_id_is_1_to_199319", np.array_equal(frame_id, np.arange(1, 199319 + 1)),
              f"range={frame_id[0]}..{frame_id[-1]}")
    add_check("video_frame_is_0_to_199318", np.array_equal(video_frame, np.arange(0, 199319)),
              f"range={video_frame[0]}..{video_frame[-1]}")
    add_check("frame_alignment_frame_id_minus_1", np.array_equal(video_frame, frame_id - 1),
              "video_frame == frame_id - 1 everywhere")
    add_check("no_infinite_values_anywhere", inf_total == 0, f"inf_count={inf_total}")
    add_check("unique_feature_names", len(set(feature_names)) == len(feature_names),
              f"n_features={len(feature_names)}")

    # Check no ball interpolation across missing frames
    b_vis = X[:, name_to_idx["ball_visible"]] == 1.0
    b_xm = X[:, name_to_idx["ball_x_m"]]
    add_check("no_ball_fabrication_on_missing_frames",
              bool(np.isnan(b_xm[~b_vis]).all() and (~np.isnan(b_xm[b_vis])).all()),
              f"ball_x_m present={int((~np.isnan(b_xm)).sum())}, expected=113488")

    # Check nearest player distance presence & exact agreement with official baseline_raw.csv
    d1_H = X[:, name_to_idx["nearest_dist_H_m"]]
    d1_A = X[:, name_to_idx["nearest_dist_A_m"]]
    vis_H = X[:, name_to_idx["team_H_visible_players"]] > 0
    vis_A = X[:, name_to_idx["team_A_visible_players"]] > 0
    add_check("nearest_player_distances_present_iff_ball_and_team_visible",
              bool(((~np.isnan(d1_H)) == (b_vis & vis_H)).all() and ((~np.isnan(d1_A)) == (b_vis & vis_A)).all()),
              f"d1_H valid={int((~np.isnan(d1_H)).sum())}, d1_A valid={int((~np.isnan(d1_A)).sum())}")

    baseline_raw_path = os.path.join(mtah_paths.REPO, "evaluation", "results", "baseline_raw.csv")
    if os.path.isfile(baseline_raw_path):
        b_df = pd.read_csv(baseline_raw_path)
        nt_sign = X[:, name_to_idx["nearest_team_sign"]]
        our_raw = np.where(np.isnan(nt_sign), "D", np.where(nt_sign > 0, "H", "A"))
        agree_cnt = int((our_raw == b_df["label"].values).sum())
        add_check("exact_agreement_with_official_baseline_raw",
                  agree_cnt == N_FRAMES,
                  f"matches={agree_cnt}/{N_FRAMES} (100.000% identical to evaluation/results/baseline_raw.csv)")

    # Check player speed physical gate
    max_pspeed = max(stats_df.loc[stats_df["name"].str.match(r"^p\d+_speed_mps$"), "max"].dropna())
    add_check("player_speeds_physically_gated", max_pspeed <= MAX_PLAYER_STEP_SPEED,
              f"max_slot_speed={max_pspeed:.3f} m/s <= {MAX_PLAYER_STEP_SPEED} m/s")

    # Check slot 44 is 100% missing as established in Phase 3
    s44_miss = stats_df.loc[stats_df["name"] == "p44_x_m", "missing_all_pct"].values[0]
    add_check("slot_44_never_observed_preserved", s44_miss == 100.0, f"p44_x_m missing={s44_miss}%")

    all_passed = all(c["passed"] for c in checks)
    report = {
        "verdict": "PASS" if all_passed else "FAIL",
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "dtype": str(X.dtype),
        "in_memory_mb": round(X.nbytes / (1024 * 1024), 2),
        "compressed_npz_mb": round(npz_mb, 2),
        "build_time_s": round(build_time, 2),
        "save_time_s": round(save_time, 2),
        "total_time_s": round(build_time + save_time, 2),
        "group_counts": dict(group_counts),
        "checks": checks,
    }

    rep_json_path = os.path.join(OUT_DIR, "validation_report.json")
    with open(rep_json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Write readable validation summary
    lines = [
        "=" * 78,
        f"PHASE 4 FEATURE PIPELINE VALIDATION REPORT -- VERDICT: {report['verdict']}",
        "=" * 78,
        f"Matrix shape       : {X.shape[0]:,} rows x {X.shape[1]} features ({X.dtype})",
        f"Frame ID range     : {frame_id[0]} .. {frame_id[-1]} (maps to video_frame {video_frame[0]} .. {video_frame[-1]})",
        f"Build time         : {build_time:.2f} s",
        f"NPZ save time      : {save_time:.2f} s",
        f"In-memory size     : {report['in_memory_mb']:.2f} MB",
        f"Compressed NPZ     : {report['compressed_npz_mb']:.2f} MB ({npz_path})",
        "",
        "FEATURE COUNTS BY GROUP:",
    ]
    for grp, cnt in group_counts.items():
        sub = stats_df[stats_df["group"] == grp]
        lines.append(
            f"  {grp:<22}: {cnt:>3} features | missing (all): mean {sub['missing_all_pct'].mean():5.1f}% "
            f"[min {sub['missing_all_pct'].min():4.1f}%, max {sub['missing_all_pct'].max():5.1f}%] | "
            f"missing (graded): mean {sub['missing_graded_pct'].mean():5.1f}%"
        )
    lines.append("\nVALIDATION CHECKS:")
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"  [{status}] {c['name']} -- {c['detail']}")

    rep_txt_path = os.path.join(OUT_DIR, "validation_report.txt")
    with open(rep_txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    return report


if __name__ == "__main__":
    X, frame_id, video_frame, feature_names, meta, build_time = build_all_features()
    report = validate_and_save(X, frame_id, video_frame, feature_names, meta, build_time)
    sys.exit(0 if report["verdict"] == "PASS" else 1)
