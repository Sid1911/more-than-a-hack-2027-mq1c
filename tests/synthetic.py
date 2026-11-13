"""A small made-up data bundle in the competition's file formats, for the tests.

Nothing here comes from the competition data: positions are drawn from a seeded random walk,
and the tag file holds a few invented phases. It only exercises the readers and the pipeline.
"""

import os

import numpy as np

FIRST, LAST = 11185, 14185   # tracked frames: two minutes inside the first graded half
N_SLOTS = 46                  # the competition ATD has 46 track slots


def write_bundle(folder, seed=0):
    tactical = os.path.join(folder, "tactical")
    os.makedirs(tactical, exist_ok=True)
    rng = np.random.default_rng(seed)

    with open(os.path.join(tactical, "BAR-ATH_tactical_atd.xml"), "w") as h:
        h.write('<main><Teams><Team id="T1"><Name>FC Barcelona</Name></Team>'
                '<Team id="T2"><Name>Athletic Club</Name></Team></Teams><Players>')
        for s in range(N_SLOTS):
            h.write('<Player id="P{}" teamId="{}"/>'.format(s, "T1" if s < 23 else "T2"))
        h.write("</Players></main>")

    pos = rng.uniform(0.1, 0.9, size=(N_SLOTS, 2))
    ball = np.array([0.5, 0.5])
    with open(os.path.join(tactical, "BAR-ATH_tactical_atd.txt"), "w") as h:
        for frame in range(FIRST, LAST):
            pos = np.clip(pos + rng.normal(0, 0.002, pos.shape), 0.0, 1.0)
            holder = (frame // 250) % 2 * 23 + 3          # possession flips every 10 s
            ball = 0.8 * ball + 0.2 * pos[holder]
            groups = []
            for s in range(N_SLOTS):
                if s % 23 < 11:
                    groups.append("{:.5f},{:.5f},{:.2f}".format(pos[s, 0], pos[s, 1], 1.0))
                else:
                    groups.append("NaN,NaN,NaN")
            b = "NaN,NaN,NaN" if frame % 7 == 0 else "{:.5f},{:.5f},0.1".format(*ball)
            h.write("{}:{}:{}\n".format(frame + 1, ";".join(groups), b))

    tags = [("SET PIECES", 447.0, 452.0, "N/A"), ("BUILD UP", 455.0, 465.0, "Barcelona"),
            ("DEFENDING IN MIDDLE THIRD", 465.0, 475.0, "Barcelona"),
            ("PROGRESSION", 475.0, 485.0, "Atletic Club"), ("GOALS", 500.0, 505.0, "Barcelona")]
    with open(os.path.join(tactical, "BAR-ATH_tactical_smart_tagging.xml"), "w") as h:
        h.write("<file><ALL_INSTANCES>")
        for i, (code, a, b, team) in enumerate(tags):
            h.write("<instance><ID>{}</ID><start>{}</start><end>{}</end><code>{}</code>"
                    "<label><group>Team</group><text>{}</text></label></instance>".format(
                        i, a, b, code, team))
        h.write("</ALL_INSTANCES></file>")
    return folder
