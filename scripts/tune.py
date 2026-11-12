"""Tune the model's settings on Metrica's public sample games -- never on the competition match.

    python scripts/tune.py --sample-dir PATH/TO/sample-data

PATH/TO/sample-data is a copy of https://github.com/metrica-sports/sample-data (games 1 and 2
are used). The sample games are degraded first (ball hidden in bursts, position noise, fake
set-piece tags) so they look like the competition's automatic tracking; see possession/simulate.py.

The search is a simple coordinate search: try each setting at a few values while holding the
others, keep the best, repeat. It maximises spellF1, averaged over both games and two random
degradations of each. Fully seeded: the same command writes the same params.json.
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import features, labels, metrics, model, simulate, tracking  # noqa: E402

PITCH = (105.0, 68.0)
SEEDS = (0, 1)

GRID = {
    "w_team": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "r_ctrl": [1.0, 1.5, 2.0, 3.0, 5.0],
    "m_scale": [1.0, 2.0, 3.0, 5.0],
    "v_ctrl": [4.0, 8.0, 15.0, 30.0],
    "b_dead": [-2.0, -1.0, -0.5, 0.0, 0.5],
    "w_act": [0.0, 0.25, 0.5, 1.0, 2.0],
    "a0": [1.0, 1.2, 1.5, 1.8, 2.2, 2.6],
    "w_tag": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "w_phase": [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    "c_switch": [5.0, 10.0, 20.0, 40.0, 80.0, 160.0],
    "c_dead": [5.0, 10.0, 20.0, 40.0, 80.0, 160.0],
}


GAMES = (1, 2, 3)


def load_game(sample_dir, k):
    folder = os.path.join(sample_dir, "data", "Sample_Game_{}".format(k))
    stem = os.path.join(folder, "Sample_Game_{}_".format(k))
    if k == 3:
        meta = stem + "metadata.xml"
        track = tracking.read_metrica_sample_epts(stem + "tracking.txt", meta, PITCH)
        home_id = [t.get("id") for t in ET.parse(meta).getroot().iter("Team")][0]
        gt, restarts = labels.from_metrica_json(stem + "events.json", track["period"],
                                                len(track["ball"]), home_id)
    else:
        track = tracking.read_metrica_sample_csv(stem + "RawTrackingData_Home_Team.csv",
                                                 stem + "RawTrackingData_Away_Team.csv", PITCH)
        gt, restarts = labels.from_metrica_events(stem + "RawEventsData.csv", track["period"],
                                                  len(track["ball"]))
    return track, gt, restarts


def degraded_cases(sample_dir):
    """[(features, gt, blocks, name)] -- each sample game under each seed."""
    cases = []
    for k in GAMES:
        track, gt, restarts = load_game(sample_dir, k)
        n = len(gt)
        blocks = []
        for p in (1, 2):
            rows = np.flatnonzero(track["period"] == p)
            blocks.append((int(rows[0]), int(rows[-1])))
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * k + seed)
            noisy = dict(track)
            noisy["players"] = simulate.jitter(track["players"], rng)
            noisy["ball"] = simulate.drop_ball(simulate.jitter(track["ball"], rng), rng)
            tag = simulate.tag_windows(restarts, n, rng)
            phases = simulate.phase_tags(gt, rng)
            cases.append((features.compute(noisy, PITCH, tag, phases), gt, blocks,
                          "game{} seed{}".format(k, seed)))
    return cases


def team_accuracy(gt, pred):
    """Share of frames that are truly H or A where the predicted team is right."""
    alive = (gt == 0) | (gt == 1)
    return float(np.mean(pred[alive] == gt[alive])) if alive.any() else 0.0


def score(cases, params):
    vals = []
    for f, gt, blocks, _ in cases:
        vals.append(metrics.spell_f1(gt, model.predict(f, params, blocks))[0])
    return float(np.mean(vals))


def baseline_like(f, blocks, window=50):
    """The organisers' smooth50 idea on the same degraded data, for comparison only."""
    n = len(f["vis"])
    code = np.full(n, -1)
    seen = f["vis"] & ~np.isnan(f["dh"]) & ~np.isnan(f["da"])
    code[seen] = np.where(f["dh"][seen] <= f["da"][seen], 0, 1)
    home = np.concatenate(([0], np.cumsum(code == 0)))
    away = np.concatenate(([0], np.cumsum(code == 1)))
    i = np.arange(n)
    lo, hi = np.maximum(0, i - window), np.minimum(n, i + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    out = np.full(n, metrics.D, dtype=np.int8)
    m = (h + a) > 0
    out[m] = np.where(h[m] >= a[m], metrics.H, metrics.A)
    final = np.full(n, metrics.X, dtype=np.int8)
    for first, last in blocks:
        final[first:last + 1] = out[first:last + 1]
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "params.json"))
    args = ap.parse_args()

    t0 = time.time()
    cases = degraded_cases(args.sample_dir)
    print("prepared {} degraded sample cases in {:.0f}s".format(len(cases), time.time() - t0))

    base = np.mean([metrics.spell_f1(gt, baseline_like(f, b))[0] for f, gt, b, _ in cases])
    print("organisers' smooth50 idea on the same cases: spellF1 {:.4f}".format(base))

    params = dict(model.DEFAULT_PARAMS)
    best = score(cases, params)
    print("starting settings: spellF1 {:.4f}".format(best))
    for r in range(args.rounds):
        improved = False
        for name, values in GRID.items():
            for v in values:
                if v == params[name]:
                    continue
                trial = dict(params, **{name: v})
                s = score(cases, trial)
                if s > best + 1e-6:
                    best, params, improved = s, trial, True
            print("round {} {:<9} = {:<6} spellF1 {:.4f}".format(r + 1, name, params[name], best))
        if not improved:
            break

    print("\nper case with the tuned settings:")
    for f, gt, blocks, name in cases:
        pred = model.predict(f, params, blocks)
        s, ng, npred = metrics.spell_f1(gt, pred)
        print("  {:<14} spellF1 {:.4f}  macroF1 {:.4f}  team acc {:.3f}  possessions true {} "
              "/ ours {}".format(name, s, metrics.macro_f1(gt, pred), team_accuracy(gt, pred),
                                 ng, npred))

    with open(args.out, "w") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)
    print("\nwrote {}  (mean spellF1 {:.4f}, {:.0f}s)".format(args.out, best, time.time() - t0))


if __name__ == "__main__":
    main()
