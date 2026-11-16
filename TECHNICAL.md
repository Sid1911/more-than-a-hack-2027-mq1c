# MQ1C (Més Que Un Code) — technical document

More than a Hack 2027, Phase 1: Ball Possession. Code: this repository.

## 1. The approach

**What we built.** A three-state model (H, A, D) that labels each half of the match in one pass.
For every frame it scores how well each state fits a few signals from the automatic tracking and
tagging. Then the Viterbi algorithm finds the single label sequence, for the whole half, with the
best total score, after paying a fixed penalty for every switch between H and A (`c_switch`) and
for every switch between live and dead (`c_dead`).

**Why this rather than something else.**

- *The metric rewards whole possessions.* spellF1 matches possessions as intervals, so ten correct
  frames split by one wrong frame score worse than ten frames that are all slightly off. A penalty
  per switch, solved exactly over the whole half, is the most direct way to produce whole
  possessions. A sliding majority vote (the organisers' baseline) has to trade flicker against lag
  with one window size; the switch penalty does not, because a switch happens exactly where the
  evidence turns, as soon as the evidence after that point is strong enough to pay for it.
- *There are no labels for this match, and few anywhere.* We may not train on the provided game,
  and the external games we found with matching tracking and possession information are Metrica's
  three public sample games. Twelve hand-chosen signals with tuned weights can be fitted reliably
  on three games. A learned classifier alone did worse than this model (0.511 against 0.538 on
  unseen games), but as a *second opinion* added to the same decoder it helps clearly (Model A,
  below).
- *The ball is missing 41% of the time.* A frame-by-frame classifier has nothing to say on those
  frames. In our model they simply carry no ball evidence, so the current state continues through
  the gap unless something else (players stopping, a restart tag) says otherwise.

**The evidence, per frame.** Distances in metres, speeds in m/s, 25 frames per second.

| signal | how it is computed | enters as |
|---|---|---|
| nearest player | distance from the ball to the nearest home player `dh` and away player `da` | `w_team · exp(−min(dh,da)/r_ctrl) · 1/(1+(v/v_ctrl)²) · clip((da−dh)/m_scale, −1, 1)`, added to H and subtracted from A. `v` is the ball's speed, so a ball in flight counts for little. Zero when the ball is not seen |
| player activity | median speed of all tracked players (positions 0.5 s apart, speeds over 12 m/s dropped as identity switches), averaged over 3 s | `w_act · clip(a0 − activity, −3, 3)` added to D: walking players mean a stoppage |
| ball outside the lines | ball seen more than 0.5 m outside the pitch | `w_out` added to D |
| set-piece tags | Smart Tagging `SET PIECES` windows | `w_tag` added to D in the first half of each window, subtracted in the second: dead before the restart, live after it |
| attacking-phase tags | Smart Tagging `BUILD UP`, `PROGRESSION`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `CREATING CHANCES`, `ATTACKING TRANSITION`, `LONG BALL`, with their Team tag | `w_phase` added to the tagged team while the phase lasts (overlaps capped at one phase's worth) |

D also gets a constant `b_dead`. Scores are rounded to 10⁻⁶ before the search so the result is
identical on every machine.

**Model A, a second opinion** (`possession/modela/`, added 16 November). A gradient-boosted tree
classifier (scikit-learn `HistGradientBoostingClassifier`, 300 trees) reads 147 measurements per
frame — ball position, speed and direction; distances, speeds and counts of players around the
ball; team shape, compactness and attacking position — and gives a probability for H, A and D.
The measurements come from `build_features.py`, written by our teammate Pranav Swaroop and used
unchanged. Its centred log-probabilities, times `w_model` = 2, are added to the scores above on
every frame where the ball is seen; where it is not seen Model A says nothing. Left out of its
inputs: Smart Tagging and homography (the training games have neither) and every
ball-visibility measurement (the sample games hide the ball at stoppages, so a model would learn
"ball missing = dead ball", which does not hold for the competition tracking). Before use, every
game is turned, from its own data, so the home goalkeeper starts on the left, as the feature
code expects. Inputs are rounded to 3 decimals so floating-point differences between machines
cannot cross a tree's split point.

**The settings** (`params.json`): `w_team` 8, `r_ctrl` 3.0 m, `v_ctrl` 8 m/s, `m_scale` 1.0 m,
`w_act` 0.5, `a0` 1.8 m/s, `w_out` 3, `w_tag` 2, `w_phase` 0.2, `b_dead` 0, `c_switch` 80,
`c_dead` 30, `w_model` 2. All but `w_out` were tuned (section 3).

## 2. Assets used and dropped

| asset | used? | why |
|---|---|---|
| tactical automatic tracking (ATD) | **yes**: player and ball x, y | the main signal. Covers the whole pitch for the whole match |
| Smart Tagging: `SET PIECES` | **yes** | restarts mark where dead-ball stretches end |
| Smart Tagging: attacking phases (7 codes above) | **yes** | the Team tag names the side in possession, by the meaning of the code |
| Smart Tagging: `DEFENDING IN …` thirds, `DEFENSIVE TRANSITION` | **no** | the documentation does not say whether their Team tag names the defending side or the attacking one, and the organisers referred us back to it. Read the wrong way they would push the model against the truth, so they are switched off (`USE_DEFENDING_TAGS` in `possession/model.py`) |
| Smart Tagging: `PLAYERS IN THE BOX`, `GOALS` | no | their meaning for who has the ball is ambiguous. `GOALS` is only used by the optional `summary.py` display, never by the submission |
| speed field in the ATD | no | we compute speeds from positions over 0.5 s, which smooths tracker jitter |
| broadcast video and broadcast tracking | no | the tactical feed already covers the whole pitch; the broadcast view shows part of it |
| tactical video, homography | no | we do not detect anything ourselves, and the rules forbid tuning on the video |

Tried and dropped:

- *Ball not seen as evidence of a dead ball.* It helped on the sample games, but only because their
  tracking blanks the ball during every stoppage — an artefact that does not carry over.
- *Learning each tag code's direction from the match's own tracking* (comparing a code's Team tag
  with the nearest-player evidence). It needed no ground truth and no video, but it fitted something
  to the provided game, so we replaced it with fixed directions from the code names.

## 3. Training data

We trained on no data from the provided game. All tuning used:

| dataset | source | labels | amount |
|---|---|---|---|
| Metrica Sports sample game 1 | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), CSV tracking + events | derived by our code from Metrica's published event data | 96 min of play, 330 possessions |
| Metrica Sports sample game 2 | same | same | 94 min, 303 possessions |
| Metrica Sports sample game 3 | same, EPTS tracking + JSON events | same | 95 min, 374 possessions |

**Who labelled it.** The events come from Metrica Sports' published sample of their event data.
We did not label anything by hand. `possession/labels.py` turns the events into one possession
label per frame using the rules in section 4.

**From clean samples to competition-like data.** The sample tracking is cleaner than the
competition's automatic tracking, so before tuning we roughen it (`possession/simulate.py`): the
ball is hidden in bursts averaging 1.5 s until 41% of frames are missing, as in the competition
data; positions get smooth noise (0.3 m); set-piece tags are faked around each restart, with 15%
missed and timing errors up to 1 s; and attacking/defending phases are faked from each possession,
with random shifts and stretches, 12% of teams flipped, 6% N/A and 10% false phases. These error
rates are our assumptions about the real tagging, not measurements of it.

**How Model A was trained** (`scripts/train_model_a.py`). On the same six roughened games, using
frames where the ball is seen, every second frame, balanced classes and a fixed seed. Its weight
was chosen on predictions for games the model never saw: for each sample game, a model trained on
the other two predicted it. Mean spellF1 over the three games was 0.534 with weight 0, rising to
0.587–0.593 for every weight from 1 to 4; we took 2, the middle of that plateau, rather than the
single best (3). The submitted model is then trained on all three games.

**How the main model was tuned.** A coordinate search over a grid for each of the 11 tuned settings, maximising
spellF1 averaged over the three games under two random roughenings each. It is seeded, so
`scripts/tune.py` writes the same `params.json` every time. `w_out` was fixed at 3 from the laws of
the game, because the sample games blank the ball whenever it is out and cannot teach it.

## 4. The possession rules we implemented

**In the training labels** (`possession/labels.py`):

- *Control.* A pass, carry, shot, recovery or set piece by a player gives his team possession from
  that moment.
- *Loose balls and challenges.* A lost ball or a challenge changes nothing on its own: possession
  stays with the last team in control until the other team shows control. A loose ball therefore
  belongs to the team that last had it.
- *Out of play and goals.* The ball is dead from the moment it goes out or a goal is scored until
  the next set piece. Touches in between — a player fetching the ball for a throw-in — do not
  restart play. If no set piece is logged within 60 s, the next controlled touch restarts play.
- *Fouls and advantage.* A foul makes the ball dead only if a set piece follows within 60 s;
  otherwise the referee played advantage and the ball stayed live.
- *Restarts.* A set piece makes the ball live and gives it to the team taking it.

**In the model** these rules show up as behaviour rather than code:

- A team change needs sustained evidence: the evidence for the new team, summed over the frames
  after the change, must exceed `c_switch` = 80. A slow ball at an opponent's feet gives up to 16
  per frame towards a switch, so a clear take-over switches within about 0.2 s; a contested ball,
  a fast ball or a missing ball gives little or nothing, so possession stays with the last team —
  the same loose-ball rule as the labels.
- A stoppage needs the players to slow down, the ball to go out, or a restart tag; play restarts
  when they speed up again or the second half of a set-piece window begins.

## 5. Results

| | spellF1 | macroF1 |
|---|---|---|
| sample games, roughened, unseen during tuning — organisers' baseline method | 0.30 | |
| same — main model, tracking only | 0.52 | |
| same — main model, tracking + attacking-phase tags | 0.534 | 0.781 |
| same — main model + Model A, each game predicted by a Model A that never saw it (submitted) | **0.591** | **0.799** |
| public 5 minutes — organisers' best baseline (smooth125) | 0.424 | 0.581 |
| public 5 minutes — main model alone (our 13 November file) | 0.526 | 0.747 |

Model A's gain held on each of the three sample games separately (+0.045 to +0.059 spellF1).
The public 5 minutes hold 17 possessions, so one possession moves spellF1 there by about 0.03.
We ran the official scorer on them only to check that our pipeline produces a valid, sensible
file, and changed no setting because of it; the decision to add Model A was made on the sample
games alone.

### Approaches we tested and did not use

All of these were judged on the sample games only, never on the provided match.

- **Training on labels made from the provided match ("pseudo-labels").** Rejected without testing:
  the clarification email requires all training and tuning to use external data only.
- **Learning the evidence weights instead of setting them by hand.** We trained a classifier
  (logistic regression, and gradient-boosted trees) on per-frame evidence from two sample games,
  fed its output into the same decoder, and scored it on the third game, rotating through all
  three. To make the comparison fair, our model was also retuned on the same two games each time.
  Average spellF1 on the held-out game: our hand-weighted model 0.538, gradient-boosted trees
  0.539, logistic regression 0.523. A gain of 0.001 is well inside the noise, and the trained
  version would add a dependency, a model file and a risk to byte-for-byte reproduction, so we
  kept the hand-weighted model.
- The same test shows our model is not overfitted to the sample games: retuned on two games, it
  scores 0.538 on the unseen third; tuned on all three, it scores 0.534.
- **Our teammates' rule-based designs.** A deterministic rulebook (contacts, possession episodes,
  eight dead-ball detectors) scored 0.113 as written and 0.241 when tuned on sample games. Model A
  on its own, with smoothing, scored 0.511. Both lost to the main model alone (0.538); Model A
  only helps as a second opinion inside it.
- **More rules on top of the main model** (16 November), each retuned jointly with all the other
  settings and judged on unseen games: a "ball under control" signal (player close and moving
  with the ball for a few frames), moving each change of team to the new team's first control
  frame, players bunching as a sign of a stoppage, and merging possessions shorter than 1–2 s.
  None gained more than 0.003; the tuning switched most of them off by itself.
- **Ball not seen as a dead-ball signal.** See section 2: an artefact of the sample data.

## 6. Limitations, and what we would fix with more time

- **Short possessions are merged.** On the sample games the model finds about 70% as many
  possessions as there are: brief touches between longer possessions are absorbed. A lower switch
  penalty would find them but would also add false switches; a switch cost that depends on the
  situation (cheap for a clear interception, dear for a contested ball) would do better.
- **Who has the ball is right about 75% of the time on the roughened samples.** The nearest player
  is a crude signal when the ball is in the air or the tracking swaps identities. A ball-trajectory
  model that follows passes through the gaps would help most.
- **Stoppages are detected late.** Player speed falls a second or two after the whistle, so dead
  stretches start late and end late.
- **Possession changes pass through a brief dead ball.** With the tuned penalties, going H → D → A
  (2 × `c_dead` = 60) is cheaper than H → A directly (`c_switch` = 80), so most changes of team show
  a dead ball shorter than a second: 853 of the 1,420 dead spells on the sample games. Removing
  them changes sample-game spellF1 by less than 0.001 (0.5338 to 0.5335), so we left the model as
  tuned. A cleaner design would forbid dead spells shorter than about a second.
- **Three practice games.** All thirteen settings and Model A rest on three matches from one
  provider, and on our guesses about how the competition's tracking and tagging differ from them.
  Model A, with 147 inputs, is the part most exposed to that difference; its evidence is capped by
  its weight and silent whenever the ball is missing, and the main model's own evidence still
  decides when the two disagree strongly.
- **Defending-phase tags are unused**, because their meaning is not documented. With them on, our
  sample-game tests give spellF1 0.554 instead of 0.540 if the tags are as reliable as we assumed,
  and 0.511 instead of 0.517 if they are no better than chance.

## 7. Reproduction

```
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # numpy, pandas, scikit-learn (all pinned)
.venv/bin/python scripts/predict.py --data-dir path/to/data/bundle
```

This writes `submission/MQ1C.csv` and prints its SHA-256: the same data and code give the same
file, byte for byte. Rebuilding `params.json`: `scripts/tune.py --sample-dir path/to/sample-data`
(about two minutes, deterministic). Rebuilding Model A and its weight:
`scripts/train_model_a.py --sample-dir path/to/sample-data` (about eight minutes). Full
instructions in [README.md](README.md).

## Rules compliance

Checked against the organisers' [RULES.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
and their clarification email of 3 November 2026, which takes precedence and permits only what it
explicitly allows.

| what we did | the rule that allows it |
|---|---|
| trained Model A on Metrica's public sample games 1–3 only | "If you need to train (or retrain), you MUST use only external data" (clarification) |
| tuned every setting on Metrica's public sample games 1–3 | "If your algorithm has parameters that need to be tuned, you MUST tune them on external data" (clarification); "External data about other games" (RULES.md) |
| used that sample data, crediting the source | "Using any dataset you have the rights to use" (clarification); the sample data is published by Metrica Sports for the community to experiment with; it has no formal licence file, and its README asks users to "be responsible with the use of this data" and to acknowledge the source in public uses, which we do |
| used the tactical tracking and the Smart Tagging as inputs | "Using as input … the automatic tracking data, the smart tagging" (clarification) |
| ran the organisers' scorer on the public 5-minute ground truth, as a pipeline check only | "Running automatic checks using only the provided data" (clarification); "for checking that your pipeline is correct" (RULES.md) |
| used an AI coding assistant to help write the code | "LLMs, hosted or local" (RULES.md). No provided data was ever uploaded to it |
| used a teammate's feature code (`possession/modela/build_features.py`) | our own team's work |
| fixed two values from football knowledge (a ball outside the lines is dead; each tag code's direction from its name) | design constants, not fitted to anything and not tied to any frame of the provided game |

What we did not do: no hand labels or manual corrections on the provided game; no fitting,
tuning or calibration on its ground truth, its video or any other data from it; no check of our
output against the video; no human-produced data about the provided game; and no match data in
this repository.
