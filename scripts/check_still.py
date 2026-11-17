"""Choose the still-ball settings (k_still, w_still) on external data, and show the check.

    python scripts/check_still.py --sample-dir PATH/TO/metrica-sample-data

The sample games blank the ball at every stoppage, so they cannot show what the model does when
a ball placed for a restart IS seen. This script therefore scores every setting under two
scenarios, both made from Metrica's public sample games 1-3 (never the competition match):

  1. as scripts/tune.py does it: ball hidden in bursts, position noise, fake tags;
  2. the same, plus a ball seen standing still at the restart spot during stoppages
     (possession/simulate.place_ball -- an assumption, labelled as such in TECHNICAL.md).

Five roughenings of each game in each scenario. Model A is retrained for each held-out game on
the other two (scenario 1, seeds 0-1, exactly as scripts/train_model_a.py trains it), so no game
is scored by a model that saw it. For each held-out game, k_still, w_still, b_dead and c_dead are
chosen on the other two games, maximising spellF1 averaged over both scenarios; the table shows
the held-out game. The last line is the choice made on all three games, which goes into
params.json. Deterministic. Takes about 15 minutes.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import train_model_a  # noqa: E402
import tune  # noqa: E402
from possession import features, metrics, model, modela, simulate  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
GRID = {"k_still": [0, 1, 2, 3, 4, 6, 8], "w_still": [0.1, 0.25, 0.5, 1.0],
        "b_dead": [-0.5, 0.0, 0.5], "c_dead": [20.0, 30.0, 40.0]}


def cases(sample_dir):
    out = []
    for k in tune.GAMES:
        track, gt, restarts = tune.load_game(sample_dir, k)
        blocks = []
        for p in (1, 2):
            rows = np.flatnonzero(track["period"] == p)
            blocks.append((int(rows[0]), int(rows[-1])))
        for scen in (1, 2):
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * k + seed)
                noisy = dict(track)
                noisy["players"] = simulate.jitter(track["players"], rng)
                ball = simulate.jitter(track["ball"], rng)
                if scen == 2:
                    ball = simulate.place_ball(ball, gt, restarts,
                                               np.random.default_rng(77 + 1000 * k + seed))
                noisy["ball"] = simulate.drop_ball(ball, rng)
                tag = simulate.tag_windows(restarts, len(gt), rng)
                phases = simulate.phase_tags(gt, rng)
                f = features.compute(noisy, tune.PITCH, tag, phases)
                out.append({"game": k, "scen": scen, "seed": seed, "f": f, "gt": gt,
                            "blocks": blocks, "X": modela.matrix(noisy, tune.PITCH, blocks)})
    return out


def dead_recall(gt, pred):
    m = gt == 2
    return float(np.mean(pred[m] == 2))


def evaluate(cs, p):
    rows = []
    for c in cs:
        pred = model.predict(c["f"], p, c["blocks"], c["ev"])
        rows.append((metrics.spell_f1(c["gt"], pred)[0], metrics.macro_f1(c["gt"], pred),
                     dead_recall(c["gt"], pred)))
    return np.array(rows)


def search(cs, p):
    best = evaluate(cs, p)[:, 0].mean()
    for _ in range(2):
        changed = False
        for name, vals in GRID.items():
            for v in vals:
                q = dict(p, **{name: v})
                s = evaluate(cs, q)[:, 0].mean()
                if s > best + 1e-9:
                    best, p, changed = s, q, True
        if not changed:
            break
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--params", default=os.path.join(ROOT, "params.json"))
    args = ap.parse_args()
    t0 = time.time()
    with open(args.params) as h:
        start = json.load(h)
    cs = cases(args.sample_dir)
    for k in tune.GAMES:
        clf = train_model_a.fit([c for c in cs if c["game"] != k and c["scen"] == 1
                                 and c["seed"] in (0, 1)])
        for c in cs:
            if c["game"] == k:
                c["ev"] = modela.evidence(clf, c["X"], c["f"]["vis"])
    for c in cs:
        c.pop("X")
    print("prepared {} cases in {:.0f}s".format(len(cs), time.time() - t0))
    before = dict(start, w_still=0.0)
    print("held-out game, scenario: spellF1 / macroF1 / dead-ball recall, before -> after")
    for k in tune.GAMES:
        p = search([c for c in cs if c["game"] != k], dict(start))
        for scen in (1, 2):
            held = [c for c in cs if c["game"] == k and c["scen"] == scen]
            a, b = evaluate(held, before).mean(0), evaluate(held, p).mean(0)
            print("game {} scenario {}: {:.3f} -> {:.3f} / {:.3f} -> {:.3f} / {:.3f} -> {:.3f}   "
                  "(chosen on the other games: {})".format(
                      k, scen, a[0], b[0], a[1], b[1], a[2], b[2],
                      {n: p[n] for n in GRID}))
    p = search(cs, dict(start))
    print("chosen on all three games (goes into params.json): {}".format(
        {n: p[n] for n in GRID}))
    print("done in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
