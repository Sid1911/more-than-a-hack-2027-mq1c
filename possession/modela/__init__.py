"""Model A: a second opinion on who has the ball, learned from external games.

Model A is a gradient-boosted tree classifier (scikit-learn HistGradientBoosting) that reads 147
per-frame measurements of the ball and the players and gives a probability for H, A and D. It
is trained only on Metrica Sports' public sample games (scripts/train_model_a.py), never on the
competition match. Its log-probabilities are added to the main model's evidence with weight
`w_model` (params.json), and the same Viterbi decoder then picks the labels.

The measurements are computed by build_features.py, written by our teammate Pranav Swaroop for
this competition and used unchanged (see layout.py for how it is run on any game). Three groups
of his features are left out:
  * Smart Tagging and homography: the external games have neither;
  * ball visibility (is the ball seen, frames since it was last seen, visibility rates, visible
    player counts): the sample games hide the ball at every stoppage, so a model would learn
    "ball missing = dead ball", which is not true of the competition tracking.
For the same reason Model A's evidence is used only on frames where the ball is seen; while the
ball is missing it says nothing and the main model decides alone.
"""

import json
import os
import pickle

import numpy as np

from . import layout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL_PATH = os.path.join(ROOT, "models", "model_a.pkl")
FEATURES_PATH = os.path.join(ROOT, "models", "model_a_features.json")
ROUND = 3  # features are rounded to 3 decimals so tiny floating-point differences between
#            machines cannot move a value across a tree's split point


def feature_names():
    with open(FEATURES_PATH) as h:
        return json.load(h)


def matrix(track, pitch, halves, names=None):
    """(n, 147) float32 matrix of Model A's inputs for a whole match.
    halves: [(first, last_inclusive)] for the two halves; the kick-offs are their first frames."""
    names = names or feature_names()
    X, all_names = layout.build(track, pitch, halves, (halves[0][0], halves[1][0]))
    idx = [all_names.index(c) for c in names]
    return np.round(X[:, idx], ROUND).astype(np.float32)


def load():
    with open(MODEL_PATH, "rb") as h:
        saved = pickle.load(h)
    import sklearn
    if saved["sklearn"] != sklearn.__version__:
        raise SystemExit("models/model_a.pkl was saved with scikit-learn {}, but {} is installed. "
                         "Install the pinned versions: pip install -r requirements.txt".format(
                             saved["sklearn"], sklearn.__version__))
    return saved["model"]


def evidence(clf, X, vis):
    """(n, 3) centred log-probabilities for H, A, D; zero where the ball is not seen."""
    lp = np.log(np.clip(clf.predict_proba(X), 1e-6, 1.0))
    lp -= lp.mean(axis=1, keepdims=True)
    lp[~np.asarray(vis, bool)] = 0.0
    return lp
