"""Run build_features.py (our teammate's feature code, unchanged) on any game.

build_features.py was written to read BAR-ATH through the organisers' loaders and `games.py`.
Here those modules are replaced by small stand-ins that serve the arrays we already have, and a
fresh copy of build_features.py is loaded on top of them. The same code path is used for the
external training games and for the competition match, so the features mean the same thing in
both.

Every game is first put into the layout build_features.py assumes:
  * 46 slots: home in 0..23 with the goalkeeper in slot 23 (slot 24 in its 1-based numbering),
    away in 24..45 with the goalkeeper in slot 45;
  * the home team attacks left to right in the first half (the whole game is rotated 180 degrees
    if not);
  * coordinates normalised to 0..1, as in the ATD.
The rotation is decided from the data itself (where the home goalkeeper stands in the first half),
the same way for every game. Smart Tagging and homography are not passed in: the stand-ins return
no tags and a zero reprojection error, and those columns are not used by Model A.
"""

import importlib.util
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_SLOTS, HOME_GK, AWAY_GK = 46, 23, 45


def to_layout(track, pitch, halves):
    """track: ball (n, 2) and players (n, S, 2) in metres, team (S,) with 0 = home.
    Returns (players_norm (n, 46, 2), ball_norm (n, 3), team_of_slot (46,))."""
    L, W = pitch
    players = np.asarray(track["players"], dtype=np.float64)
    ball = np.asarray(track["ball"], dtype=np.float64)
    team = np.asarray(track["team"])
    h0, h1 = halves[0]
    first = slice(h0, h1 + 1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_x = np.nanmean(players[first, :, 0], axis=0)
    # rotate so that home attacks left to right in the first half (its goalkeeper is on the left)
    home_cols = np.flatnonzero(team == 0)
    away_cols = np.flatnonzero(team == 1)
    home_gk = home_cols[np.nanargmin(np.abs(mean_x[home_cols] - 0.0))] if np.isfinite(
        mean_x[home_cols]).any() else home_cols[0]
    if mean_x[home_gk] > L / 2.0:
        players = np.stack([L - players[..., 0], W - players[..., 1]], axis=-1)
        ball = np.stack([L - ball[:, 0], W - ball[:, 1]], axis=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_x = np.nanmean(players[first, :, 0], axis=0)
    home_gk = home_cols[np.nanargmin(np.where(np.isfinite(mean_x[home_cols]), mean_x[home_cols],
                                              np.inf))]
    away_gk = away_cols[np.nanargmax(np.where(np.isfinite(mean_x[away_cols]), mean_x[away_cols],
                                              -np.inf))]
    n = len(ball)
    out = np.full((n, N_SLOTS, 2), np.nan, dtype=np.float32)
    team46 = np.array([0] * 24 + [1] * 22)
    # if a team has more tracked slots than the layout holds, keep the most-tracked ones (an
    # unused substitute's slot is empty; an active player's is not)
    tracked = np.sum(~np.isnan(players[:, :, 0]), axis=0)
    home_rest = sorted([c for c in home_cols if c != home_gk], key=lambda c: -tracked[c])[:23]
    away_rest = sorted([c for c in away_cols if c != away_gk], key=lambda c: -tracked[c])[:21]
    home_rest, away_rest = sorted(home_rest), sorted(away_rest)
    for i, c in enumerate(home_rest):
        out[:, i] = players[:, c]
    out[:, HOME_GK] = players[:, home_gk]
    for i, c in enumerate(away_rest):
        out[:, 24 + i] = players[:, c]
    out[:, AWAY_GK] = players[:, away_gk]
    out[..., 0] /= L
    out[..., 1] /= W
    b = np.full((n, 3), np.nan, dtype=np.float32)
    b[:, 0], b[:, 1] = ball[:, 0] / L, ball[:, 1] / W
    return out, b, team46


def build(track, pitch, halves, kickoffs):
    """All 587 build_features.py columns for an external game. halves: [(first, last)] frames,
    kickoffs: (first-half kickoff frame, second-half kickoff frame). Returns (X, names)."""
    players, ball, team46 = to_layout(track, pitch, halves)
    n = len(ball)
    stubs = {
        "mtah_paths": types.SimpleNamespace(configure=lambda: None),
        "games": types.SimpleNamespace(
            GAME="EXTERNAL", GAMES={"EXTERNAL": {"fps": 25, "pitch": tuple(pitch)}},
            source_frames=lambda source, game=None: n,
            half_video_blocks=lambda game=None: [(1, halves[0][0], halves[0][1]),
                                                 (2, halves[1][0], halves[1][1])]),
        "load_atd": types.SimpleNamespace(atd_paths=lambda *a, **k: ("meta", "raw")),
        "load_homography": types.SimpleNamespace(
            load=lambda *a, **k: (None, np.zeros(n), np.ones(n, dtype=bool))),
        "load_smart_tagging": types.SimpleNamespace(
            KICKOFF_VIDEO_FRAMES={"1st Half": int(kickoffs[0]), "2nd Half": int(kickoffs[1])},
            load=lambda *a, **k: []),
        "baseline": types.SimpleNamespace(team_of_each_track=lambda meta: team46,
                                          team_names=lambda meta: ("HOME", "AWAY")),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("build_features_external",
                                                      os.path.join(HERE, "build_features.py"))
        bf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bf)
        frame_id = np.arange(1, n + 1, dtype=np.int64)
        bf.load_cached_atd = lambda: (frame_id, players, ball, team46, "meta")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            X, _, _, names, _, _ = bf.build_all_features()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return X, [str(x) for x in names]
