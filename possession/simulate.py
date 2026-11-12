"""Make clean sample-game tracking look like the competition's automatic tracking.

The sample games come from corrected tracking: the ball is almost always there. The
competition ATD loses the ball on about 41% of graded frames, in bursts, and its tactical
tagging is automatic and imperfect. Tuning on clean data would pick settings that lean on a
ball the competition data often does not have, so we degrade the sample games first.

Everything here is seeded, so tuning is reproducible run to run.
"""

import numpy as np

FPS = 25


def drop_ball(ball, rng, missing_share=0.41, mean_gap_s=1.5):
    """Hide the ball in random bursts: a two-state chain, visible <-> missing."""
    n = len(ball)
    p_return = 1.0 / (mean_gap_s * FPS)  # chance per frame of a missing ball coming back
    p_lose = p_return * missing_share / (1.0 - missing_share)
    hidden = np.zeros(n, dtype=bool)
    draws = rng.random(n)
    missing = False
    for i in range(n):
        missing = (draws[i] >= p_return) if missing else (draws[i] < p_lose)
        hidden[i] = missing
    out = ball.copy()
    out[hidden] = np.nan
    return out


def jitter(xy, rng, sigma_m=0.3, smooth_frames=12):
    """Position noise in metres, smooth over time as tracker error is (it drifts, it does not
    jump independently every frame). White noise averaged over ~0.5 s, rescaled to sigma_m."""
    white = rng.normal(0.0, 1.0, size=xy.shape)
    csum = np.concatenate([np.zeros((1,) + xy.shape[1:]), np.cumsum(white, axis=0)], axis=0)
    k = smooth_frames
    smooth = (csum[k:] - csum[:-k]) / np.sqrt(k)  # unit variance again
    noise = np.concatenate([smooth, np.repeat(smooth[-1:], k - 1, axis=0)], axis=0)
    return (xy + sigma_m * noise).astype(np.float32)


ATTACK_CODES = ("BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "BALL IN THE BOX",
                "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL")
DEFEND_CODES = ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")


def phase_tags(labels, rng, cover=0.7, flip=0.12, na=0.06, spurious=0.1, defend_share=0.45):
    """Fake tactical phases for a sample game, shaped like the competition's Smart Tagging.

    The real tagging has ~580 phases over a match, each an interval with a code and a Team tag.
    Attacking codes (Build Up, Progression...) name the team WITH the ball and defending codes
    (Defending in ...) the team WITHOUT it; model.PHASE_DIRECTION fixes that from the names.
    Here each true possession spell gets, with probability `cover`, one phase that is shifted and
    stretched at random, carries an attacking or defending code, has its team flipped with
    probability `flip` and set to N/A with probability `na`. A share `spurious` of extra phases
    land anywhere with a random team.

    These rates are assumptions about the real tagging, not measurements of it -- the real file
    is never inspected for this. Returns [(first, last_exclusive, code, team)], team -1 = N/A.
    """
    n = len(labels)
    out = []
    edges = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [n]))
    for a, b in zip(starts, ends):
        team = int(labels[a])
        if team not in (0, 1) or b - a < 2 * FPS or rng.random() > cover:
            continue
        length = (b - a) * rng.uniform(0.5, 1.2)
        start = a + rng.uniform(-2, 2) * FPS
        defend = rng.random() < defend_share
        code = DEFEND_CODES[rng.integers(len(DEFEND_CODES))] if defend \
            else ATTACK_CODES[rng.integers(len(ATTACK_CODES))]
        tagged = (1 - team) if defend else team
        if rng.random() < flip:
            tagged = 1 - tagged
        if rng.random() < na:
            tagged = -1
        first = int(max(0, start))
        out.append((first, int(min(n, first + length)), code, tagged))
    for _ in range(int(spurious * len(out))):
        first = int(rng.integers(0, n))
        codes = ATTACK_CODES + DEFEND_CODES
        out.append((first, min(n, first + int(rng.uniform(3, 12) * FPS)),
                    codes[rng.integers(len(codes))], int(rng.integers(0, 2))))
    return out


def tag_windows(restarts, n, rng, half_window_s=2.5, miss_rate=0.15, shift_s=1.0):
    """Fake set-piece tags: a window around each true restart, some missed, some shifted.

    The real Smart Tagging marks set pieces as ~5 s windows around the restart, so the fake
    ones have the same shape. Returns a 0/1 mask; features.tag_evidence turns it into a
    before/after signal exactly as it does for the real tags.
    """
    tag = np.zeros(n, dtype=bool)
    half = int(half_window_s * FPS)
    for frame in restarts:
        if rng.random() < miss_rate:
            continue
        centre = int(frame + rng.uniform(-shift_s, shift_s) * FPS)
        tag[max(0, centre - half):min(n, centre + half)] = True
    return tag
