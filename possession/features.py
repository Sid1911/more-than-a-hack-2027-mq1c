"""Per-frame evidence, computed once per match.

Nothing here decides anything; the model in model.py weighs this evidence.

    vis       ball seen this frame
    out       ball seen outside the pitch lines (dead ball, by the laws of the game)
    dh, da    distance from the ball to the nearest home / away player, metres
    bspeed    ball speed, m/s (NaN when the ball is not seen)
    activity  how fast the players are moving: median player speed, smoothed over ~3 s.
              Players jog and sprint while the ball is live and walk during stoppages.
    tag       set-piece windows from the tactical tagging, as a signed signal: +1 in the
              first half of each window (just before the restart: ball dead), -1 in the
              second half (just after it: ball live), 0 elsewhere.
    phases    tactical phases from the tagging: [(first, last_exclusive, code, team)], team
              0 / 1, or -1 for N/A. Which way each code points is worked out in model.py.
"""

import warnings

import numpy as np

FPS = 25
MAX_PLAYER_SPEED = 12.0  # m/s; anything faster is a tracker identity jump, not a person
PLAYER_SPEED_SPAN = 6    # speeds over +/- 6 frames (~0.5 s), so tracker jitter averages out
BALL_SPEED_SPAN = 2


def rolling_mean(x, half):
    """Centred mean over +/- half frames, ignoring NaN."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    sums = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    counts = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    total, count = sums[hi] - sums[lo], counts[hi] - counts[lo]
    out = np.full(len(x), np.nan)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out


def span_speed(xy, span):
    """|x[t+span] - x[t-span]| / (2 span / fps); NaN near the ends or where either is missing."""
    speed = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    step = np.sqrt(((xy[2 * span:] - xy[:-2 * span]) ** 2).sum(-1)) * (FPS / (2.0 * span))
    speed[span:-span] = step
    return speed


def tag_evidence(mask):
    """Signed restart signal from a 0/1 window mask: +1 first half of each window, -1 second."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(len(mask))
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        mid = (a + b) // 2
        out[a:mid] = 1.0
        out[mid:b] = -1.0
    return out


def compute(track, pitch, tag_mask=None, phases=None):
    ball, players, team = track["ball"], track["players"], track["team"]
    length, width = pitch
    n = len(ball)

    vis = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        out = vis & ((ball[:, 0] < -0.5) | (ball[:, 0] > length + 0.5)
                     | (ball[:, 1] < -0.5) | (ball[:, 1] > width + 0.5))

    dist = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))  # (n, S), NaN where missing
    home = team == 0
    dh = np.fmin.reduce(np.where(home[None, :], dist, np.nan), axis=1)
    da = np.fmin.reduce(np.where(~home[None, :], dist, np.nan), axis=1)

    bspeed = span_speed(ball, BALL_SPEED_SPAN).astype(np.float64)
    bspeed[~vis] = np.nan

    pspeed = span_speed(players, PLAYER_SPEED_SPAN)
    pspeed[pspeed > MAX_PLAYER_SPEED] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_speed = np.nanmedian(pspeed, axis=1)
    activity = rolling_mean(median_speed, int(1.5 * FPS))

    return {
        "vis": vis, "out": out, "dh": dh, "da": da, "bspeed": bspeed, "activity": activity,
        "tag": np.zeros(n) if tag_mask is None else tag_evidence(tag_mask),
        "phases": list(phases or []),
    }
