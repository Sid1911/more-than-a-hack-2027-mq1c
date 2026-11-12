"""Read tracking data into one common shape, whatever file it came from.

Every reader returns a dict:
    ball     (n, 2) float32, metres from the top-left corner, NaN where the ball is not seen
    players  (n, S, 2) float32, metres, NaN where a slot is empty
    team     (S,) int8, 0 = home, 1 = away
    period   (n,) int8, 1 or 2, 0 outside the halves (only filled by the sample-game reader)

Row i is frame i counted from 0. The competition ATD numbers its frames from 1, so a line with
frame_id f lands in row f - 1, which is also row f - 1 of the submission.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def _team_of_slots(meta_path):
    """0/1 per track slot, in the order the slots appear in the raw file.

    Same rule as the organisers' baseline: the first <Team> in the metadata is the home side,
    and the <Player> entries are listed in slot order.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def read_epts(raw_path, meta_path, pitch, n_frames, cache_path=None):
    """The competition ATD (FIFA/EPTS raw text + metadata XML).

    Line format: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<ball fields>". Only x and y are
    used; coordinates are 0-1 and are scaled to metres here. Parsing 199k lines takes about a
    minute, so the result is cached as .npz next to the data when cache_path is given.
    """
    source = np.array([os.path.getsize(raw_path), os.path.getsize(meta_path), n_frames],
                      dtype=np.int64)
    if cache_path and os.path.isfile(cache_path):
        z = np.load(cache_path)
        if "source" in z.files and np.array_equal(z["source"], source):
            return {k: z[k] for k in z.files if k != "source"}
        # cache from another file, or from an older version of this code: read again

    team = _team_of_slots(meta_path)
    n_slots = len(team)
    length, width = pitch
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) < 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            groups = parts[1].split(";")[:n_slots]
            xy = []
            for group in groups:
                fields = group.split(",")
                xy.append(float(fields[0]))
                xy.append(float(fields[1]) if len(fields) > 1 else np.nan)
            players[row, :len(groups)] = np.array(xy, dtype=np.float32).reshape(-1, 2)
            fields = parts[2].split(",")
            if len(fields) >= 2:
                ball[row] = (float(fields[0]), float(fields[1]))

    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    out = {"ball": ball, "players": players, "team": team,
           "period": np.zeros(n_frames, dtype=np.int8)}
    if cache_path:
        np.savez(cache_path, source=source, **out)
    return out


def read_metrica_sample_epts(raw_path, meta_path, pitch=(105.0, 68.0)):
    """Metrica's public sample game 3 (FIFA EPTS, the same family as the competition ATD).

    Unlike the competition file, the slot order changes with every substitution: the metadata has
    one <DataFormatSpecification> per frame range, each listing which player is in which slot.
    Every frame is re-ordered here into a fixed layout -- home players first, then away -- so the
    rest of the code can keep one team label per slot.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    channel_player = {ch.get("id"): ch.get("playerId") for ch in root.iter("PlayerChannel")}
    player_team = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    period_bounds = {}
    for param in root.iter("ProviderParameter"):
        name, value = param.findtext("Name"), param.findtext("Value")
        if name and value and name.endswith(("half_start", "half_end")):
            period_bounds[name] = int(value)

    specs = []  # (first_frame, last_frame, [team index per slot])
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        slots = [0 if player_team[channel_player[r]] == team_ids[0] else 1
                 for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots))

    per_team = 11
    n = max(last for _, last, _ in specs)
    players = np.full((n, 2 * per_team, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    length, width = pitch
    spec_i = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) < 3:
                continue
            frame = int(parts[0])
            while spec_i < len(specs) - 1 and frame > specs[spec_i][1]:
                spec_i += 1
            slots = specs[spec_i][2]
            counts = [0, 0]
            for slot, group in zip(slots, parts[1].split(";")):
                fields = group.split(",")
                if len(fields) < 2 or counts[slot] >= per_team:
                    continue
                players[frame - 1, slot * per_team + counts[slot]] = (float(fields[0]),
                                                                    float(fields[1]))
                counts[slot] += 1
            fields = parts[2].split(",")
            if len(fields) >= 2:
                ball[frame - 1] = (float(fields[0]), float(fields[1]))

    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    period = np.zeros(n, dtype=np.int8)
    for p, key in ((1, "first"), (2, "second")):
        a = period_bounds.get(key + "_half_start")
        b = period_bounds.get(key + "_half_end")
        if a and b:
            period[a - 1:b] = p
    team = np.array([0] * per_team + [1] * per_team, dtype=np.int8)
    return {"ball": ball, "players": players, "team": team, "period": period}


def read_metrica_sample_csv(home_csv, away_csv, pitch=(105.0, 68.0)):
    """Metrica's public sample games 1 and 2 (CSV, one file per team).

    The first two rows are team and shirt headers; the third names the columns. Each player
    has an x column followed by an unnamed y column, and the ball pair comes last.
    """
    length, width = pitch
    tables = []
    for path in (home_csv, away_csv):
        frame = pd.read_csv(path, skiprows=2)
        tables.append(frame)
    home, away = tables
    period = home["Period"].to_numpy().astype(np.int8)
    frames = home["Frame"].to_numpy()
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError("expected frames numbered 1..n in {}".format(home_csv))

    def xy_block(frame):
        values = frame.iloc[:, 3:].to_numpy(dtype=np.float64)
        return values.reshape(len(frame), -1, 2)  # last pair is the ball

    home_xy, away_xy = xy_block(home), xy_block(away)
    ball = home_xy[:, -1, :].copy()
    players = np.concatenate([home_xy[:, :-1, :], away_xy[:, :-1, :]], axis=1)
    team = np.array([0] * (home_xy.shape[1] - 1) + [1] * (away_xy.shape[1] - 1), dtype=np.int8)

    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return {"ball": ball.astype(np.float32), "players": players.astype(np.float32),
            "team": team, "period": period}
