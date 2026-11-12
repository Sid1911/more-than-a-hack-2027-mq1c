# data/

Empty on purpose: the match data is licensed for the competition only and must not be committed.

Unpack the organisers' data bundle here so that these three files exist:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

The three files placed directly in `data/` work too, as does pointing `MTAH_DATA_DIR` (or
`--data-dir`) at the bundle wherever it already is. Nothing else from the bundle is used: not the
videos, not the broadcast tracking, not the homography.

On the first run `scripts/predict.py` writes `BAR-ATH_tactical_atd_cache.npz` next to the tracking
file (about 75 MB) so later runs skip the one-minute parse. It is rebuilt automatically if the
tracking file changes, and it is ignored by git.
