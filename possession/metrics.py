"""spellF1 and macroF1, re-implemented to the organisers' definitions for tuning on OTHER games.

The official scorer (evaluation/possession_metrics.py in the challenge repository) remains the
authority; this copy exists so the tuning script needs nothing outside this package. It uses
the same rules: possessions are maximal H or A runs inside each graded stretch, matched one to
one by temporal IoU (greedy, highest IoU first), F1 averaged over IoU 0.3 to 0.7. Predicted X
is forward-filled before matching, exactly as the official scorer does.
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def graded_runs(gt):
    edges = np.diff(np.concatenate(([0], (gt != X).astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fill_x(pred, runs):
    out = pred.copy()
    for a, b in runs:
        w = out[a:b]
        known = np.flatnonzero(w != X)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(w != X, np.arange(len(w)), -1))
        carry[carry < 0] = known[0]
        out[a:b] = w[carry]
    return out


def spells(codes, runs):
    out = []
    for a, b in runs:
        w = codes[a:b]
        if not len(w):
            continue
        cuts = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        for i in range(len(cuts) - 1):
            k = int(w[cuts[i]])
            if k in (H, A):
                out.append((a + int(cuts[i]), a + int(cuts[i + 1]), k))
    return out


def spell_f1(gt, pred):
    runs = graded_runs(gt)
    pred = fill_x(pred, runs)
    g, p = spells(gt, runs), spells(pred, runs)
    pairs, j0 = [], 0
    for i, (gs, ge, gk) in enumerate(g):          # both lists are in time order
        while j0 < len(p) and p[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < len(p) and p[j][0] < ge:
            ps, pe, pk = p[j]
            if pk == gk:
                inter = min(ge, pe) - max(gs, ps)
                union = max(ge, pe) - min(gs, ps)
                pairs.append((inter / float(union), i, j))
            j += 1
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    f1s = []
    for thr in IOUS:
        used_g, used_p, tp = set(), set(), 0
        for iou, i, j in pairs:
            if iou < thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            tp += 1
        prec = tp / len(p) if p else 0.0
        rec = tp / len(g) if g else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)), len(g), len(p)


def macro_f1(gt, pred):
    m = gt != X
    g, p = gt[m], pred[m]
    f1s = []
    for k in (H, A, D):
        tp = np.sum((g == k) & (p == k))
        fp = np.sum((g != k) & (p == k))
        fn = np.sum((g == k) & (p != k))
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))
