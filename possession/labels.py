"""Possession labels for OTHER games, derived from their human event data.

Used only to tune the model's settings on Metrica's public sample games. The rules forbid
tuning on the competition match, so the competition ground truth never enters this code.

Codes everywhere in this package: 0 = H, 1 = A, 2 = D (dead ball), 3 = X (no label).

The derivation follows the challenge's definition of possession:
  * a set piece starts play and gives the ball to the team taking it;
  * a pass, carry, shot or recovery means that team has control;
  * a lost ball or a challenge changes nothing until the other side shows control;
  * a ball going out or a goal makes the ball dead until the next set piece; touches in between
    (a player fetching the ball) do not restart play. If no restart is logged within 60 seconds,
    the next controlled touch restarts play instead;
  * a foul makes the ball dead only if a set piece follows within 60 seconds -- otherwise the
    referee played advantage and the ball stayed live.
"""

import json

import numpy as np
import pandas as pd

H, A, D, X = 0, 1, 2, 3
CONTROL = {"PASS", "SHOT", "RECOVERY", "SET PIECE", "CARRY"}
FPS = 25
MAX_DEAD_S = 60  # how soon a restart must follow for a stoppage to count as one


def _from_table(ev, period, n_frames):
    """ev: DataFrame with Period, Type, Subtype, Team ('home'/'away'), Start Frame, End Frame
    (1-based frames). Returns per-frame codes and the restart frames."""
    ev = ev.sort_values(["Period", "Start Frame"], kind="stable")
    labels = np.full(n_frames, X, dtype=np.int8)
    restarts = []

    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        if not len(rows):
            continue
        first, last = int(rows[0]), int(rows[-1]) + 1
        state, cursor = None, first

        def advance(to, new_state):
            # Paint the current state up to frame `to`, then switch to new_state.
            nonlocal state, cursor
            to = min(max(to, cursor), last)
            if state is not None:
                labels[cursor:to] = state
            cursor, state = to, new_state

        half_ev = ev[ev["Period"] == p]
        set_piece_frames = np.sort(
            half_ev.loc[half_ev["Type"].str.upper() == "SET PIECE", "Start Frame"].to_numpy() - 1)

        def restart_within(frame):
            """True if a set piece is logged within MAX_DEAD_S after `frame`."""
            i = np.searchsorted(set_piece_frames, frame)
            return i < len(set_piece_frames) and set_piece_frames[i] - frame <= MAX_DEAD_S * FPS

        hold_dead = False  # a stoppage confirmed by a following restart: ignore touches until it
        for _, e in half_ev.iterrows():
            kind = str(e["Type"]).upper()
            sub = str(e["Subtype"]).upper()
            team = H if str(e["Team"]).strip().lower() == "home" else A
            start = int(e["Start Frame"]) - 1
            end = int(e["End Frame"]) - 1 if e["End Frame"] > 0 else start

            if kind == "SET PIECE":
                advance(start, team)
                restarts.append(start)
                hold_dead = False
            elif kind in CONTROL:
                # During a stoppage, a player fetching or carrying the ball is not play.
                if hold_dead:
                    continue
                advance(start, team)
                if kind == "SHOT" and "GOAL" in sub:
                    advance(end, D)
                    hold_dead = True
            elif kind == "BALL OUT":
                # Out of play is always dead. Touches before the restart are ignored only when
                # the restart is actually logged soon after; otherwise the next touch restarts.
                advance(max(end, start), D)
                hold_dead = restart_within(max(end, start))
            elif kind == "FAULT RECEIVED":
                # A foul stops play only when a free kick follows; without one, the referee
                # played advantage and the ball never went dead.
                if restart_within(start):
                    advance(start, D)
                    hold_dead = True
            # BALL LOST, CHALLENGE, CARD: no change of state.
        advance(last, state)

    return labels, np.array(sorted(restarts), dtype=np.int64)


def from_metrica_events(events_csv, period, n_frames):
    """Sample games 1 and 2 (CSV events)."""
    return _from_table(pd.read_csv(events_csv), period, n_frames)


def from_metrica_json(events_json, period, n_frames, home_team_id):
    """Sample game 3 (JSON events). CARRY exists only in this format and counts as control."""
    rows = []
    for e in json.load(open(events_json))["data"]:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = sub.get("name")
        elif isinstance(sub, list):
            sub = " ".join(s.get("name", "") for s in sub)
        rows.append({
            "Period": e["period"],
            "Type": e["type"]["name"],
            "Subtype": sub or "",
            "Team": "home" if e["team"]["id"] == home_team_id else "away",
            "Start Frame": e["start"]["frame"],
            "End Frame": (e.get("end") or {}).get("frame") or e["start"]["frame"],
        })
    return _from_table(pd.DataFrame(rows), period, n_frames)
