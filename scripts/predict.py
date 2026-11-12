"""Write the submission file for the competition match.

    python scripts/predict.py                     # data in ./data, file in ./submission/
    python scripts/predict.py --data-dir PATH     # or set MTAH_DATA_DIR

Reads the tactical ATD and the Smart Tagging, applies the settings in params.json (tuned on
other games by scripts/tune.py), and writes 199,319 rows of frame,label. Deterministic: the
same data and params.json always give the same file, and the SHA-256 it prints lets anyone
check that their copy matches ours byte for byte.
"""

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import features, model, tracking  # noqa: E402

# Published in the challenge's DATA.md and games.py.
N_FRAMES = 199319
PITCH = (105.3, 68.0)
HALVES = [(11185, 90651), (110808, 190197)]  # graded video frames, inclusive
STEM = "BAR-ATH_tactical"
FPS = 25
with open(os.path.join(ROOT, "team.txt")) as _handle:  # one line: our registered team name
    TEAM = _handle.read().strip()  # the submission file must be named <team name>.csv
LETTERS = np.array(["H", "A", "D", "X"])


def home_is_barcelona(meta_path):
    """True if the first <Team> in the tracking metadata -- the side the organisers' code and the
    labels call home (H) -- is Barcelona. Read from the team names; falls back to the match name
    BAR-ATH (home team first) if the metadata carries no recognisable name."""
    names = [(t.findtext("Name") or t.get("name") or "").lower()
             for t in ET.parse(meta_path).getroot().iter("Team")]
    if any("barcelona" in name for name in names):
        return "barcelona" in names[0]
    print("   note: no team names in the tracking metadata; assuming Barcelona is home (BAR-ATH)")
    return True


def read_tagging(path, n, barcelona_home):
    """From the Smart Tagging: a mask of SET PIECES windows, and every instance as a phase
    (first, last_exclusive, code, team). Team 0 = home, 1 = away, -1 = N/A."""
    tag = np.zeros(n, dtype=bool)
    phases = []
    for inst in ET.parse(path).iter("instance"):
        code = (inst.findtext("code") or "").strip().upper()
        a = max(0, int(round(float(inst.findtext("start")) * FPS)))
        b = min(n, int(round(float(inst.findtext("end")) * FPS)))
        team_name = ""
        for label in inst.findall("label"):
            if (label.findtext("group") or "").strip().lower() == "team":
                team_name = (label.findtext("text") or "").strip().lower()
        if team_name in ("", "n/a"):
            team = -1
        else:
            team = 0 if (("barcelona" in team_name) == barcelona_home) else 1
        if code == "SET PIECES":
            tag[a:b] = True
        phases.append((a, b, code, team))
    return tag, phases


def main():
    ap = argparse.ArgumentParser()
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--data-dir", help="the unpacked data bundle (default: MTAH_DATA_DIR, "
                                          "else ./data); its tactical/ folder is used")
    where.add_argument("--tactical-dir", help="a folder holding the BAR-ATH_tactical_* files")
    ap.add_argument("--params", default=os.path.join(ROOT, "params.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "submission", TEAM + ".csv"))
    args = ap.parse_args()

    if args.tactical_dir:
        folder = args.tactical_dir
    else:
        data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
        folder = os.path.join(data_dir, "tactical")
        if not os.path.isdir(folder):
            folder = data_dir  # the three files placed directly in the data folder
    raw = os.path.join(folder, STEM + "_atd.txt")
    meta = os.path.join(folder, STEM + "_atd.xml")
    tags = os.path.join(folder, STEM + "_smart_tagging.xml")
    missing = [p for p in (raw, meta, tags) if not os.path.isfile(p)]
    if missing:
        sys.exit("These match data files are missing:\n  " + "\n  ".join(missing))

    with open(args.params) as handle:
        params = json.load(handle)

    print("reading tracking (first run takes about a minute, then it is cached) ...")
    track = tracking.read_epts(raw, meta, PITCH, N_FRAMES,
                               cache_path=os.path.join(folder, STEM + "_atd_cache.npz"))
    print("computing evidence ...")
    barcelona_home = home_is_barcelona(meta)
    print("home side (H) in the tracking: {}".format(
        "Barcelona" if barcelona_home else "Athletic Club"))
    tag_mask, phases = read_tagging(tags, N_FRAMES, barcelona_home)
    used = sum(1 for _, _, code, team in phases
               if team in (0, 1) and (model.PHASE_DIRECTION.get(code, 0.0) > 0
                                      or (model.USE_DEFENDING_TAGS
                                          and model.PHASE_DIRECTION.get(code, 0.0) < 0)))
    print("tactical phases used: {} of {}".format(used, len(phases)))
    f = features.compute(track, PITCH, tag_mask, phases)
    print("labelling ...")
    codes = model.predict(f, params, HALVES)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(args.out, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, c) for i, c in enumerate(LETTERS[codes]))

    graded = np.zeros(N_FRAMES, dtype=bool)
    for a, b in HALVES:
        graded[a:b + 1] = True
    g = codes[graded]
    spells = sum(1 for a, b in HALVES for i in range(a, b + 1)
                 if codes[i] in (0, 1) and (i == a or codes[i - 1] != codes[i]))
    with open(args.out, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print("wrote {}".format(args.out))
    print("graded frames: H {:.1%}  A {:.1%}  D {:.1%}   possessions: {}".format(
        np.mean(g == 0), np.mean(g == 1), np.mean(g == 2), spells))
    print("SHA-256 {}  (the same on every machine that reproduces our file)".format(digest))


if __name__ == "__main__":
    main()
