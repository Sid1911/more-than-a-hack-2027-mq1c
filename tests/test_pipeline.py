"""Tests. Run from the repository root:

    python -m unittest discover -s tests -v

They use a small made-up data bundle (tests/synthetic.py), never the competition data, so they
can run anywhere, including on a machine that has not received the data.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synthetic  # noqa: E402
from possession import labels, metrics, model  # noqa: E402

HALVES = [(11185, 90651), (110808, 190197)]
N_ROWS = 199319


def predict(data_dir, out, *extra):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "predict.py"), "--data-dir", data_dir,
           "--out", out] + list(extra)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    if proc.returncode != 0:
        raise AssertionError("predict.py failed:\n" + proc.stdout)
    return proc.stdout


def read_bytes(path):
    with open(path, "rb") as h:
        return h.read()


def write_bytes(path, data):
    with open(path, "wb") as h:
        h.write(data)


def sha256(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mtah-test-")
        cls.bundle = synthetic.write_bundle(os.path.join(cls.tmp, "bundle"))
        cls.out = os.path.join(cls.tmp, "a.csv")
        cls.log = predict(cls.bundle, cls.out, "--no-cache")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_submission_format(self):
        raw = read_bytes(self.out)
        raw.decode("ascii")                                   # plain ASCII
        self.assertNotIn(b"\r", raw)                          # LF newlines only
        self.assertTrue(raw.endswith(b"\n"))
        lines = raw.decode().split("\n")[:-1]
        self.assertEqual(lines[0], "frame,label")
        self.assertEqual(len(lines) - 1, N_ROWS)
        frames = [int(l.split(",")[0]) for l in lines[1:]]
        self.assertEqual(frames, list(range(N_ROWS)))         # every frame, in order
        lab = np.array([l.split(",")[1] for l in lines[1:]])
        graded = np.zeros(N_ROWS, dtype=bool)
        for a, b in HALVES:
            graded[a:b + 1] = True
        self.assertTrue(set(lab[graded]) <= {"H", "A", "D"})  # a label on every graded frame
        self.assertTrue((lab[~graded] == "X").all())          # X outside the halves

    def test_prints_its_checksum_and_home_side(self):
        self.assertIn("SHA-256 " + sha256(self.out), self.log)
        self.assertIn("home side (H) in the tracking: Barcelona", self.log)
        self.assertIn("Model A used with weight", self.log)

    def test_deterministic_with_and_without_cache(self):
        b = os.path.join(self.tmp, "b.csv")
        c = os.path.join(self.tmp, "c.csv")
        predict(self.bundle, b)        # writes the cache
        predict(self.bundle, c)        # reads the cache
        self.assertEqual(sha256(self.out), sha256(b))
        self.assertEqual(sha256(b), sha256(c))

    def test_stale_cache_is_ignored(self):
        bundle = synthetic.write_bundle(os.path.join(self.tmp, "bundle2"))
        first = os.path.join(self.tmp, "first.csv")
        predict(bundle, first)                                # cache of the original file
        fresh = synthetic.write_bundle(os.path.join(self.tmp, "fresh"), seed=1)
        shutil.copy(os.path.join(fresh, "tactical", "BAR-ATH_tactical_atd.txt"),
                    os.path.join(bundle, "tactical", "BAR-ATH_tactical_atd.txt"))
        with open(os.path.join(bundle, "tactical", "BAR-ATH_tactical_atd.txt"), "a") as h:
            h.write("\n")                                     # different size than the cached one
        after = os.path.join(self.tmp, "after.csv")
        predict(bundle, after)
        expected = os.path.join(self.tmp, "expected.csv")
        predict(bundle, expected, "--no-cache")
        self.assertEqual(sha256(after), sha256(expected))

    def test_verify_script(self):
        verify = os.path.join(ROOT, "scripts", "verify_submission.py")
        same = subprocess.run([sys.executable, verify, self.out, self.out],
                              stdout=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(same.returncode, 0, same.stdout)
        self.assertIn("RESULT: IDENTICAL", same.stdout)
        changed = os.path.join(self.tmp, "changed.csv")
        lines = read_bytes(self.out).decode().split("\n")
        lines[20000] = lines[20000][:-1] + ("A" if lines[20000].endswith("H") else "H")
        write_bytes(changed, "\n".join(lines).encode())
        diff = subprocess.run([sys.executable, verify, self.out, changed],
                              stdout=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(diff.returncode, 1)
        self.assertIn("label value", diff.stdout)
        crlf = os.path.join(self.tmp, "crlf.csv")
        write_bytes(crlf, read_bytes(self.out).replace(b"\n", b"\r\n"))
        nl = subprocess.run([sys.executable, verify, self.out, crlf],
                            stdout=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(nl.returncode, 1)
        self.assertIn("newlines differs", nl.stdout)


class LabelRuleTests(unittest.TestCase):
    """The possession rules used to label the external sample games."""

    def label(self, events, n=3000):
        import pandas as pd
        ev = pd.DataFrame(events, columns=["Period", "Type", "Subtype", "Team", "Start Frame",
                                           "End Frame"])
        period = np.ones(n, dtype=np.int8)
        return labels._from_table(ev, period, n)[0]

    def test_out_of_play_is_dead_until_the_restart(self):
        out = self.label([(1, "SET PIECE", "KICK OFF", "Home", 1, 1),
                          (1, "PASS", "", "Home", 50, 60),
                          (1, "BALL OUT", "", "Home", 500, 510),
                          (1, "CARRY", "", "Away", 600, 650),     # fetching the ball
                          (1, "SET PIECE", "THROW IN", "Away", 900, 900),
                          (1, "PASS", "", "Away", 920, 930)])
        self.assertTrue((out[100:509] == labels.H).all())
        self.assertTrue((out[509:899] == labels.D).all())       # the carry did not restart play
        self.assertTrue((out[899:1000] == labels.A).all())

    def test_loose_ball_stays_with_last_team(self):
        out = self.label([(1, "SET PIECE", "KICK OFF", "Home", 1, 1),
                          (1, "BALL LOST", "", "Home", 100, 100),
                          (1, "CHALLENGE", "", "Away", 150, 150),
                          (1, "RECOVERY", "", "Away", 300, 300)])
        self.assertTrue((out[0:299] == labels.H).all())
        self.assertTrue((out[299:400] == labels.A).all())

    def test_foul_with_advantage_is_not_dead(self):
        out = self.label([(1, "SET PIECE", "KICK OFF", "Home", 1, 1),
                          (1, "FAULT RECEIVED", "", "Home", 200, 200),
                          (1, "PASS", "", "Home", 210, 220)])
        self.assertFalse((out == labels.D).any())

    def test_foul_with_free_kick_is_dead(self):
        out = self.label([(1, "SET PIECE", "KICK OFF", "Home", 1, 1),
                          (1, "FAULT RECEIVED", "", "Home", 200, 200),
                          (1, "SET PIECE", "FREE KICK", "Home", 700, 700)])
        self.assertTrue((out[199:699] == labels.D).all())


class ModelTests(unittest.TestCase):
    def test_phase_directions(self):
        n = 400
        ev = model.phase_evidence([(0, 100, "BUILD UP", 0), (100, 200, "PROGRESSION", 1),
                                   (200, 300, "DEFENDING IN MIDDLE THIRD", 0),
                                   (300, 400, "PLAYERS IN THE BOX", 1)], n)
        self.assertEqual(ev[50], 1.0)                 # home builds up: home has the ball
        self.assertEqual(ev[150], -1.0)               # away progresses: away has the ball
        self.assertEqual(ev[250], 0.0)                # defending codes are switched off
        self.assertEqual(ev[350], 0.0)                # ambiguous codes are not used
        self.assertFalse(model.USE_DEFENDING_TAGS)

    def test_switch_penalty_gives_whole_spells(self):
        rng = np.random.default_rng(0)
        truth = np.repeat([0, 1, 0, 2, 1], 400)
        e = rng.normal(0, 1.0, (len(truth), 3))
        e[np.arange(len(truth)), truth] += 1.5         # noisy evidence for the truth
        # c_switch below 2 * c_dead, so a direct H<->A switch is cheaper than a detour through D
        path = model.viterbi(e, 50.0, 30.0)
        switches = int(np.sum(path[1:] != path[:-1]))
        self.assertEqual(switches, 4)                  # one switch per real change, no flicker
        self.assertGreater(np.mean(path == truth), 0.95)

    def test_still_ball_switches_off_team_evidence(self):
        from possession import features
        n = 400
        ball = np.zeros((n, 2), dtype=np.float32)
        ball[:, 0] = np.where(np.arange(n) < 200, 50.0, 50.0 + 0.2 * (np.arange(n) - 200))
        ball[:, 1] = 34.0                               # still for 8 s, then rolling at 5 m/s
        players = np.stack([ball + 1.0, ball + 30.0], axis=1)  # home player 1.4 m away
        track = {"ball": ball, "players": players, "team": np.array([0, 1])}
        f = features.compute(track, (105.0, 68.0))
        self.assertGreater(f["still"][100], 0.99)      # placed ball
        self.assertEqual(f["still"][350], 0.0)         # ball in play
        p = dict(model.DEFAULT_PARAMS, w_still=0.1, k_still=4.0)
        e = model.emissions(f, p)
        self.assertEqual(e[100, 0], 0.0)               # no team evidence at a placed ball
        self.assertGreater(e[100, 2], e[100, 0])       # it leans to dead ball instead
        off = model.emissions(f, dict(p, w_still=0.0))
        self.assertGreater(off[100, 0], 0.0)           # switched off: team evidence as before

    def test_params_file(self):
        with open(os.path.join(ROOT, "params.json")) as h:
            params = json.load(h)
        self.assertEqual(set(params), set(model.DEFAULT_PARAMS))

    def test_spell_f1_of_identical_files_is_one(self):
        gt = np.repeat(np.array([0, 1, 2, 0], dtype=np.int8), 300)
        self.assertEqual(metrics.spell_f1(gt, gt.copy())[0], 1.0)


class RepositoryTests(unittest.TestCase):
    @unittest.skipIf(shutil.which("git") is None or not os.path.isdir(os.path.join(ROOT, ".git")),
                     "not a git checkout")
    def test_audit_passes(self):
        proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "audit_repo.py")],
                              stdout=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(proc.returncode, 0, proc.stdout)


if __name__ == "__main__":
    unittest.main()
