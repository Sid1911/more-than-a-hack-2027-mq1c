# MQ1C (Més Que Un Code) — More than a Hack 2027, Phase 1: Ball Possession

Our entry to [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027):
for every frame of FC Barcelona vs Athletic Club, who has the ball — home (`H`), away (`A`) or
nobody, because the ball is dead (`D`).

No deep learning and no labels from the provided game. A three-state model weighs a handful of
signals from the automatic tracking and tagging, then picks the one label sequence per half that
keeps possessions whole — because whole possessions are what the ranking metric counts.

## The approach in one screen

| | |
|---|---|
| Input | the tactical automatic tracking (players + ball) and the Smart Tagging. Nothing else |
| Output | `submission/MQ1C.csv`, 199,319 rows of `frame,label` |
| Method | per-frame evidence for H / A / D, then a Viterbi search with a penalty for every switch |
| Evidence | nearest player to the ball · how fast the players move · ball outside the lines · set-piece tags · attacking-phase tags |
| Tuning | 12 settings, tuned only on Metrica's three public sample games, roughened to look like the competition's tracking |
| Training labels | none from the provided game. Sample-game labels come from Metrica's published event data |
| Runtime | about 1 minute on a laptop (plus about 1 minute the first time, to parse the tracking) |
| Reproducible | deterministic; the run prints the file's SHA-256 so you can check it matches ours |

The full write-up — approach, assets used and dropped, training data, possession rules,
limitations — is in [TECHNICAL.md](TECHNICAL.md).

## Reproduce our submission file

Python 3.9 or newer (we used 3.12). Unpack the organisers' data bundle into `./data` (see
[data/README.md](data/README.md)), then:

```
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt          # Windows: .venv\Scripts\python
.venv/bin/python scripts/predict.py
```

That writes `submission/MQ1C.csv` and prints its SHA-256. The bundle can stay where it is:
`MTAH_DATA_DIR=/path/to/bundle` or `--data-dir /path/to/bundle` work too. On Linux or macOS,
`./reproduce.sh` runs all three steps. On Windows, double-click `run.bat`: it does the same, and if
the organisers' repository sits next to this one as `mtah\`, it also validates the file and scores
it on the public 5 minutes.

Validate before submitting, with the organisers' own checker:

```
python path/to/more-than-a-hack-2027/evaluation/validate_submission.py submission/MQ1C.csv
```

### Check the code and the file (optional)

```
.venv/bin/python -m unittest discover -s tests -v                  # tests on made-up data, no match data needed
.venv/bin/python scripts/verify_submission.py ours.csv theirs.csv  # byte-for-byte comparison of two files
.venv/bin/python scripts/reproduce_check.py --data-dir path/to/bundle --original submission/MQ1C.csv
.venv/bin/python scripts/audit_repo.py                             # no match data anywhere in the repo or its history
```

`reproduce_check.py` clones the repository into an empty folder, installs the pinned packages in a
new environment, regenerates the file from scratch and compares it with the original byte for byte.

### Regenerate the settings (optional)

`params.json` is the output of a deterministic search. To rebuild it, clone
[metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) and run:

```
.venv/bin/python scripts/tune.py --sample-dir path/to/sample-data
```

It takes about two minutes and writes the same `params.json` every time.

## What is in this repository

```
possession/
  tracking.py     reads the competition tracking and the three sample games into one shape
  features.py     per-frame evidence: distances, speeds, player activity, tags
  model.py        the three-state model, the tactical-code directions and the Viterbi search
  labels.py       possession labels for the sample games, from their event data
  simulate.py     roughens the sample games to look like the competition's automatic tracking
  metrics.py      spellF1 and macroF1, re-implemented to the organisers' definitions, for tuning
scripts/
  predict.py      writes the submission file      <- the command that regenerates our file
  tune.py         rebuilds params.json from the sample games
  summary.py      turns a submission into a readable list of possessions and goals
  verify_submission.py   compares two submission files byte for byte
  reproduce_check.py     fresh clone + clean install + regenerate + compare
  audit_repo.py          checks every file and every commit for match data before publishing
tests/            tests on a small made-up data bundle (synthetic.py), never the match data
params.json       the tuned settings
team.txt          our team name, used to name the submission file
run.bat           one-click run on Windows; demo.bat shows the possessions
reproduce.sh      one-command run on Linux / macOS
data/             empty: the match data goes here and is never committed
TECHNICAL.md      the technical document
```

## Results

Ranking score (spellF1) on Metrica's three sample games, roughened like the competition data and
never seen during tuning (average of three games × two roughenings):

| | spellF1 |
|---|---|
| organisers' baseline method (nearest player + majority vote over 2 or 5 s) | 0.30 |
| our model, tracking only | 0.52 |
| our model, tracking + attacking-phase tags (submitted) | 0.54 |

On the organisers' public 5 minutes, see [TECHNICAL.md](TECHNICAL.md#results). That window holds
17 possessions, so it is a pipeline check, not a benchmark — and we never tuned on it.

## Rules

We follow the organisers' [RULES.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
and their clarification email of 3 November 2026. In short: no hand labels and no tuning on the
provided game, its ground truth or its video; no output checked against the video; and the match
data was never given to any AI service. An AI coding assistant helped write this code; it never saw
the match data. Details in [TECHNICAL.md](TECHNICAL.md#rules-compliance).

## Credits

- Challenge, scorer and competition data: Metrica Sports, FC Barcelona Innovation Hub and Mobile
  World Capital Barcelona.
- Tuning data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data), games
  1–3 — used with thanks, as its README asks public uses to acknowledge the source.

## Licence

MIT for the code and documentation in this repository — see [LICENSE](LICENSE). The match data is
not in this repository and is not covered by it.
