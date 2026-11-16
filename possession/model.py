"""Three-state sticky model: H, A or D on every frame, chosen for the whole half at once.

Each frame gets a score for each state from the evidence (emissions). Switching state costs a
fixed penalty. The Viterbi algorithm then finds the single label sequence with the best total
score. The penalties are what stop flicker: a switch only happens when the evidence for the
new state, summed over the frames that follow, outweighs the cost of switching. That is what
the ranking metric rewards -- whole possessions, not individual frames.

When the ball is not seen, the team evidence is zero, so the model simply holds the current
state through the gap unless something else (players stopping, a set-piece tag) says the
ball went dead.
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3

DEFAULT_PARAMS = {
    # team evidence, when the ball is seen
    "w_team": 1.0,     # weight of "who is nearest the ball"
    "r_ctrl": 2.0,     # metres; evidence fades as the nearest player gets farther away
    "m_scale": 3.0,    # metres; distance gap between teams that counts as decisive
    "v_ctrl": 8.0,     # m/s; a fast ball is in flight, so proximity means little
    # dead-ball evidence
    "b_dead": -0.5,    # baseline preference against D
    "w_out": 3.0,      # ball outside the lines. FIXED, not tuned: the sample games blank the
                       # ball when it is out, so they cannot teach this. Set from the laws of
                       # the game -- a ball over the line is dead.
    "w_act": 0.5,      # per m/s of player activity below a0
    "a0": 1.8,         # m/s; typical median player speed at the live/dead boundary
    "w_tag": 0.5,      # set-piece tag: dead just before the restart, live just after
    "w_phase": 0.0,    # tactical phase tags: which team the phase says has the ball
    # Model A (possession/modela): weight of its log-probabilities; 0 = not used
    "w_model": 0.0,
    # switching penalties
    "c_switch": 30.0,  # H <-> A
    "c_dead": 30.0,    # alive <-> D, either way
}


def emissions(f, p):
    """(n, 3) scores for H, A, D."""
    n = len(f["vis"])
    e = np.zeros((n, 3))

    dmin = np.fmin(f["dh"], f["da"])
    close = np.exp(-np.nan_to_num(dmin, nan=1e9) / p["r_ctrl"])
    gate = 1.0 / (1.0 + (np.nan_to_num(f["bspeed"], nan=0.0) / p["v_ctrl"]) ** 2)
    with np.errstate(invalid="ignore"):
        margin = np.clip((f["da"] - f["dh"]) / p["m_scale"], -1.0, 1.0)
    team = p["w_team"] * close * gate * np.nan_to_num(margin, nan=0.0)
    team[~f["vis"]] = 0.0
    team = team + p.get("w_phase", 0.0) * phase_evidence(f["phases"], n)
    e[:, H] = team
    e[:, A] = -team

    activity = np.nan_to_num(f["activity"], nan=p["a0"])
    slow = np.clip(p["a0"] - activity, -3.0, 3.0)
    # "Ball not seen" is deliberately NOT evidence of a dead ball. The sample games used for
    # tuning blank the ball during every stoppage, so a weight on it would be learned from an
    # artefact of that data and would not transfer to the competition tracking.
    e[:, D] = (p["b_dead"] + p["w_act"] * slow + p["w_out"] * f["out"]
               + p["w_tag"] * f["tag"])
    return e


# Which way each Smart Tagging code points, fixed from what the code's name means in football.
# +1: the Team tag names the team WITH the ball. -1: it names the team WITHOUT it.
# Codes not listed (PLAYERS IN THE BOX, SET PIECES, GOALS, anything unknown) are ignored: their
# meaning for possession is ambiguous. Nothing here is learned from the competition match.
PHASE_DIRECTION = {
    "BUILD UP": 1.0,
    "PROGRESSION": 1.0,
    "CREATING CHANCES": 1.0,
    "BALL IN FINAL THIRD": 1.0,
    "BALL IN THE BOX": 1.0,
    "ATTACKING TRANSITION": 1.0,
    "LONG BALL": 1.0,
    "DEFENDING IN DEFENSIVE THIRD": -1.0,
    "DEFENDING IN MIDDLE THIRD": -1.0,
    "DEFENDING IN ATTACKING THIRD": -1.0,
    "DEFENSIVE TRANSITION": -1.0,  # the team that has just lost the ball
}


# The organisers' documentation does not say whether the Team tag on a "DEFENDING ..." phase names
# the defending side (our reading) or the attacking side, and the organisers referred us back to the
# documentation. So the defending codes are switched off: if our reading were wrong they would push
# the model the wrong way, while the attacking codes mean the same under either reading.
USE_DEFENDING_TAGS = False


def phase_evidence(phases, n):
    """+1 where the phases say home has the ball, -1 for away, summed over overlapping phases and
    capped at +/-1 so a pile-up of overlapping tags cannot outvote everything else."""
    out = np.zeros(n)
    for first, last, code, tagged in phases:
        if tagged not in (0, 1):
            continue
        direction = PHASE_DIRECTION.get(code, 0.0)
        if direction < 0 and not USE_DEFENDING_TAGS:
            continue
        out[max(0, first):min(n, last)] += direction * (1.0 if tagged == 0 else -1.0)
    return np.clip(out, -1.0, 1.0)


def viterbi(e, c_switch, c_dead):
    """Best path through H/A/D given per-frame scores and switching costs."""
    n = len(e)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    eh, ea, ed = e[:, 0].tolist(), e[:, 1].tolist(), e[:, 2].tolist()
    back = bytearray(3 * n)
    sh, sa, sd = eh[0], ea[0], ed[0]
    for t in range(1, n):
        # into H: from H (free), A (switch), D (restart)
        a, b, c = sh, sa - c_switch, sd - c_dead
        if a >= b and a >= c:
            nh, kh = a, 0
        elif b >= c:
            nh, kh = b, 1
        else:
            nh, kh = c, 2
        # into A
        a, b, c = sh - c_switch, sa, sd - c_dead
        if b >= a and b >= c:
            na, ka = b, 1
        elif a >= c:
            na, ka = a, 0
        else:
            na, ka = c, 2
        # into D
        a, b, c = sh - c_dead, sa - c_dead, sd
        if c >= a and c >= b:
            nd, kd = c, 2
        elif a >= b:
            nd, kd = a, 0
        else:
            nd, kd = b, 1
        sh, sa, sd = nh + eh[t], na + ea[t], nd + ed[t]
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = kh, ka, kd

    path = np.empty(n, dtype=np.int8)
    state = int(np.argmax([sh, sa, sd]))
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[3 * t + state]
    return path


def predict(f, p, blocks, model_a=None):
    """Labels for a whole match. blocks = [(first, last_inclusive), ...]; X elsewhere.
    model_a: optional (n, 3) evidence from Model A (possession.modela.evidence), weighted by
    p["w_model"]."""
    n = len(f["vis"])
    labels = np.full(n, X, dtype=np.int8)
    e = emissions(f, p)
    if model_a is not None and p.get("w_model", 0.0):
        e = e + p["w_model"] * model_a
    # Rounded to 1e-6 so the decision is the same on every machine: numpy's vectorised exp can
    # differ in the last bit between CPUs, which could flip a near-tie in the search below.
    e = np.round(e, 6)
    for first, last in blocks:
        labels[first:last + 1] = viterbi(e[first:last + 1], p["c_switch"], p["c_dead"])
    return labels
