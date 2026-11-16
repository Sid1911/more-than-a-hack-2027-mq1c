"""Train Model A on external data only, and choose its weight in the decoder.

    python scripts/train_model_a.py --sample-dir PATH/TO/metrica-sample-data

Uses Metrica Sports' public sample games 1-3 (https://github.com/metrica-sports/sample-data),
each roughened twice exactly as scripts/tune.py does (same seeds), so the model sees tracking as
patchy as the competition's. Labels come from each game's own event data (possession/labels.py).
Nothing here touches the competition match.

1. Weight: for each sample game, a model trained on the other two predicts it (so every
   prediction is on a game the model never saw). The weight w_model with the best mean spellF1
   over the three games is printed; put it in params.json.
2. Model: trained on all three games and saved to models/model_a.pkl.

Training uses frames where the ball is seen (see possession/modela/__init__.py), every 2nd frame,
balanced classes, a fixed seed and one thread, so the saved model is the same on every run.
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import tune  # noqa: E402
from possession import features, metrics, model, modela, simulate  # noqa: E402

WEIGHTS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def cases(sample_dir):
    """Same roughened games and seeds as tune.degraded_cases, plus Model A's inputs."""
    out = []
    names = modela.feature_names()
    for k in tune.GAMES:
        track, gt, restarts = tune.load_game(sample_dir, k)
        n = len(gt)
        blocks = []
        for p in (1, 2):
            rows = np.flatnonzero(track["period"] == p)
            blocks.append((int(rows[0]), int(rows[-1])))
        for seed in tune.SEEDS:
            rng = np.random.default_rng(1000 * k + seed)
            noisy = dict(track)
            noisy["players"] = simulate.jitter(track["players"], rng)
            noisy["ball"] = simulate.drop_ball(simulate.jitter(track["ball"], rng), rng)
            tag = simulate.tag_windows(restarts, n, rng)
            phases = simulate.phase_tags(gt, rng)
            f = features.compute(noisy, tune.PITCH, tag, phases)
            X = modela.matrix(noisy, tune.PITCH, blocks, names)
            out.append({"game": k, "f": f, "gt": gt, "blocks": blocks, "X": X})
    return out


def fit(train):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs, ys = [], []
    for c in train:
        m = np.zeros(len(c["gt"]), bool)
        for a, b in c["blocks"]:
            m[a:b + 1] = True
        m &= (c["gt"] <= 2) & c["f"]["vis"]
        rows = np.flatnonzero(m)[::2]
        Xs.append(c["X"][rows])
        ys.append(c["gt"][rows])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                         class_weight="balanced", early_stopping=False,
                                         random_state=0)
    return clf.fit(np.vstack(Xs), np.concatenate(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--params", default=os.path.join(ROOT, "params.json"))
    args = ap.parse_args()
    t0 = time.time()
    with open(args.params) as h:
        params = json.load(h)
    cs = cases(args.sample_dir)
    print("features for {} roughened games in {:.0f}s".format(len(cs), time.time() - t0))

    held = []
    for k in tune.GAMES:
        clf = fit([c for c in cs if c["game"] != k])
        held += [(c, modela.evidence(clf, c["X"], c["f"]["vis"])) for c in cs if c["game"] == k]
    table = {}
    for w in WEIGHTS:
        p = dict(params, w_model=w)
        preds = [(c["gt"], model.predict(c["f"], p, c["blocks"], ev)) for c, ev in held]
        table[w] = (float(np.mean([metrics.spell_f1(g, q)[0] for g, q in preds])),
                    float(np.mean([metrics.macro_f1(g, q) for g, q in preds])))
        print("w_model {:<5} spellF1 {:.4f}  macroF1 {:.4f}   (each game predicted by a model "
              "that never saw it)".format(w, *table[w]))
    best = max(WEIGHTS, key=lambda w: table[w][0])
    print("best w_model = {}".format(best))

    clf = fit(cs)
    import sklearn
    os.makedirs(os.path.dirname(modela.MODEL_PATH), exist_ok=True)
    with open(modela.MODEL_PATH, "wb") as h:
        pickle.dump({"model": clf, "sklearn": sklearn.__version__,
                     "features": modela.feature_names(), "trained_on": "Metrica sample games "
                     "1-3, two roughenings each, ball-seen frames, every 2nd frame"}, h, protocol=4)
    print("wrote {} ({:.0f}s)".format(modela.MODEL_PATH, time.time() - t0))


if __name__ == "__main__":
    main()
