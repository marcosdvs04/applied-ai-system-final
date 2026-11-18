# Evaluation results

**16 of 16 checks passed.** Average confidence on answered requests: 0.78 (min 0.30, max 1.00).

| Test input | Evaluation criterion | Status | Confidence | Top song | Result |
|---|---|---|---|---|---|
| `music for studying` | Top song is lofi | ok | 0.75 | Focus Flow | Pass |
| `I'm at the gym` | Top song energy >= 0.9 | ok | 0.75 | Gym Hero | Pass |
| `party tonight` | Top song is edm | ok | 0.75 | Neon Euphoria | Pass |
| `rainy sad afternoon` | Top song is folk or classical | ok | 1.00 | Hollow Pines | Pass |
| `night drive` | Top song is synthwave | ok | 1.00 | Night Drive Loop | Pass |
| `coffee shop vibes` | Top song is jazz | ok | 0.88 | Coffee Shop Stories | Pass |
| `I am so angry` | Top song is metal | ok | 0.82 | Iron Furnace | Pass |
| `romantic dinner date` | Top song is r&b | ok | 1.00 | Velvet Promise | Pass |
| `reggaeton para bailar` | Top song is reggaeton | ok | 0.82 | Calor de Medianoche | Pass |
| `jazz` | Handles genre-only input: top is jazz | low_confidence | 0.30 | Coffee Shop Stories | Pass |
| `edm with energy 1.5` | Out-of-range energy clamped to 1.0 | ok | 0.55 | Neon Euphoria | Pass |
| `k-pop` | Unknown genre flagged, no crash | needs_clarification | 0.00 | - | Pass |
| `asdf qwer zxcv` | Gibberish -> asks for clarification | needs_clarification | 0.00 | - | Pass |
| `` | Empty input -> rejected | rejected | 0.00 | - | Pass |
| `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...` | Oversized input -> rejected | rejected | 0.00 | - | Pass |
| `study music, no acoustic` | Explicit 'no acoustic' overrides KB hint | ok | 0.79 | Focus Flow | Pass |
