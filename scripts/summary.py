r"""Turn a submission file (199,319 rows of frame,label) into a readable list of possessions.

    python scripts/summary.py PATH\TO\ours.csv

Prints, for each half, every possession with its match clock and length, then totals:
possession count, average length, each side's share of live time and dead-ball time.
Also writes <name>_possessions.csv next to the input, one row per possession, for Excel.

It reads only our own output file -- no match data.
"""

import argparse
import csv
import os
import sys

FPS = 25
HALVES = [(11185, 90651), (110808, 190197)]  # graded video frames, inclusive (games.py)
HALF_START_MIN = [0, 45]
NAMES = {"H": "Barcelona", "A": "Athletic Club", "D": "dead ball"}


def clock(frame, half):
    seconds = (frame - HALVES[half][0]) / float(FPS) + HALF_START_MIN[half] * 60
    return "{:02d}:{:02d}".format(int(seconds // 60), int(seconds % 60))


def half_of(frame):
    for half, (first, last) in enumerate(HALVES):
        if first <= frame <= last:
            return half
    return None


def goals_from_tags(path):
    """GOALS instances from the Smart Tagging: [(first_frame, last_frame, team_name)].

    The tagging is automatic (Metrica's own, unreviewed), so a goal can be missing or misplaced.
    Read on this machine only; nothing is sent anywhere."""
    import xml.etree.ElementTree as ET
    out = []
    for inst in ET.parse(path).iter("instance"):
        if (inst.findtext("code") or "").strip().upper() != "GOALS":
            continue
        team = "?"
        for label in inst.findall("label"):
            if (label.findtext("group") or "").strip().lower() == "team":
                team = (label.findtext("text") or "?").strip()
        if "barcelona" in team.lower():
            team = NAMES["H"]
        elif team.lower() not in ("n/a", "?", ""):
            team = NAMES["A"]
        out.append((int(round(float(inst.findtext("start")) * FPS)),
                    int(round(float(inst.findtext("end")) * FPS)), team))
    return sorted(out)


def print_goals(path, labels):
    goals = goals_from_tags(path)
    print("\nGoals (from the organisers' automatic tags)")
    if not goals:
        print("  none tagged")
        return
    score = {NAMES["H"]: 0, NAMES["A"]: 0}
    for first, last, team in goals:
        half = half_of(last) if half_of(last) is not None else half_of(first)
        when = clock(last, half) if half is not None else "frame {}".format(last)
        if team in score:
            score[team] += 1
        # who our model said had the ball in the last live frames before the goal
        ours = None
        for i in range(min(last, len(labels) - 1), max(first, 0) - 1, -1):
            if labels[i] in ("H", "A"):
                ours = NAMES[labels[i]]
                break
        check = "" if ours is None else ("  (our model: {} on the ball just before - {})".format(
            ours, "matches" if ours == team else "differs"))
        print("  around {}  {:<14} {}-{}{}".format(when, team, score[NAMES["H"]], score[NAMES["A"]],
                                                   check))
    print("  final score by the tags: {} {} - {} {}".format(
        NAMES["H"], score[NAMES["H"]], score[NAMES["A"]], NAMES["A"]))


def runs(labels, first, last):
    """Maximal runs of one label inside [first, last]: (start, end_inclusive, label)."""
    out, start = [], first
    for i in range(first + 1, last + 2):
        if i > last or labels[i] != labels[start]:
            out.append((start, i - 1, labels[start]))
            start = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--show", type=int, default=20, help="possessions to print per half (0 = all)")
    ap.add_argument("--tagging", help="BAR-ATH_tactical_smart_tagging.xml, to list the goals")
    args = ap.parse_args()

    with open(args.submission) as handle:
        reader = csv.reader(handle)
        next(reader)
        labels = [row[1] for row in reader]
    if len(labels) < HALVES[-1][1] + 1:
        sys.exit("expected at least {} rows, found {}".format(HALVES[-1][1] + 1, len(labels)))

    out_path = os.path.splitext(args.submission)[0] + "_possessions.csv"
    rows = []
    frames = {"H": 0, "A": 0, "D": 0}
    for half, (first, last) in enumerate(HALVES):
        print("\n{} half".format("First" if half == 0 else "Second"))
        print("  {:<13} {:<14} {:>7}".format("match time", "who", "length"))
        shown = 0
        for a, b, lab in runs(labels, first, last):
            seconds = (b - a + 1) / float(FPS)
            frames[lab] = frames.get(lab, 0) + (b - a + 1)
            rows.append([half + 1, clock(a, half), clock(b + 1, half), NAMES.get(lab, lab),
                         round(seconds, 1), a, b])
            if args.show == 0 or shown < args.show:
                print("  {}-{}   {:<14} {:>5.1f} s".format(clock(a, half), clock(b + 1, half),
                                                         NAMES.get(lab, lab), seconds))
                shown += 1
        total = sum(1 for r in rows if r[0] == half + 1)
        if args.show and total > args.show:
            print("  ... and {} more rows (all of them are in the spreadsheet)".format(
                total - args.show))

    spells = [r for r in rows if r[3] in (NAMES["H"], NAMES["A"])]
    live = frames["H"] + frames["A"]
    print("\nWhole match")
    print("  possessions found:        {}".format(len(spells)))
    print("  average possession:       {:.1f} s".format(sum(r[4] for r in spells) / max(1, len(spells))))
    print("  Barcelona share of play:  {:.0%}".format(frames["H"] / float(max(1, live))))
    print("  Athletic share of play:   {:.0%}".format(frames["A"] / float(max(1, live))))
    print("  ball dead:                {:.0f} min".format(frames["D"] / float(FPS) / 60))

    with open(out_path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["half", "from", "to", "who", "seconds", "first_frame", "last_frame"])
        w.writerows(rows)
    if args.tagging:
        if os.path.isfile(args.tagging):
            print_goals(args.tagging, labels)
        else:
            print("\n(goals not shown: {} not found)".format(args.tagging))

    print("\nwrote {}".format(out_path))


if __name__ == "__main__":
    main()
