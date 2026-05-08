# Allsvenskan Predictor

Bayesian football prediction engine for Swedish league matches, built with Kedro + CmdStanPy.

The project trains a Poisson goal model on historical match results (with time decay), predicts the next round of fixtures, and simulates the rest of the season table.

## Pipeline at a glance

1. Load historical matches.
2. Filter to Allsvenskan (`SE1`).
3. Apply exponential time-decay weights.
4. Map team names to team IDs.
5. Build Stan input data and sample posterior draws.
6. Load and validate upcoming fixtures.
7. Predict match-level probabilities for the coming round.
8. Run Monte Carlo season simulations for standings probabilities.
9. Export JSON outputs to data and docs folders.

# Math behind this

The model is a Bayesian Poisson goals model with team attack/defense strengths and a home advantage term.

For match $m$ with home team $h$ and away team $a$:

$$
G_{h,m} \sim \text{Poisson}(\lambda_{h,m}), \quad
G_{a,m} \sim \text{Poisson}(\lambda_{a,m})
$$

$$
\log \lambda_{h,m} = \text{attack}_h - \text{defense}_a + \gamma
$$

$$
\log \lambda_{a,m} = \text{attack}_a - \text{defense}_h
$$

Where:

- $\text{attack}_t$ is team $t$'s attacking strength
- $\text{defense}_t$ is team $t$'s defensive strength
- $\gamma$ is league-wide home advantage

### Time decay

Older matches are down-weighted with exponential decay:

$$
w_i = e^{-\xi \cdot \text{ageYears}_i}
$$

Current default is $\xi = 0.7$.

### Match outcome probabilities

Given posterior draws of $(\lambda_h, \lambda_a)$:

- Home win: $P(G_h > G_a)$
- Draw: $P(G_h = G_a)$
- Away win: $P(G_h < G_a)$

These are computed via the Skellam distribution over goal difference.

### Scoreline matrix

For each scoreline $i$-$j$ (default $0..6$ goals each side):

$$
P(i,j) = P(G_h=i) \cdot P(G_a=j)
$$

Then averaged across posterior draws.

### Seasonal simulation

For each simulation run:

1. Start from current table points/goal records from played matches.
2. For each remaining fixture, sample one posterior draw.
3. Sample home/away goals from Poisson rates.
4. Apply points rules (3/1/0) and tie-break order:
points, goal difference, goals for, team name.

After many runs (`n_simulations`), convert frequencies into:

- per-position probabilities
- title probability
- top-3 probability
- relegation probability
- expected points and quantiles

# What it outputs

Two JSON artifacts are written on each run:

## 1) Coming-round match predictions

Path:

- `data/07_model_output/coming_predictions.json`
- `docs/coming_predictions.json` (copied for publishing/consumption)

Each row contains:

- fixture identity (`date`, `home_team`, `away_team`)
- expected goals (`lambda_home`, `lambda_away`)
- outcome probabilities (`p_home_win`, `p_draw`, `p_away_win`)
- `scoreline_probs` for 0-0 through 6-6
- unknown team flags (`unknown_team`, `unknown_home`, `unknown_away`)
- competition metadata (`competition_format`)

If competition format is `knockout`, it also includes:

- `p_home_advance`
- `p_away_advance`

## 2) Seasonal table projection

Path:

- `data/07_model_output/seasonal_predictions.json`
- `docs/seasonal_predictions.json` (copied for publishing/consumption)

Top-level fields include:

- `league`, `season`
- `simulations`, `random_seed`
- `rounds_completed`, `min_upcoming_round`, `max_upcoming_round`, `total_remaining_rounds`
- `teams` array

Each `teams` item includes:

- `team`
- `position_probs` (position -> probability)
- `p_champion`, `p_top3`, `p_relegation`
- `expected_points`, `points_p10`, `points_p90`

# What Data it needs

## Historical matches CSV

Configured by `matches_path` (default: `data/01_raw/swedish_leagues_2011_2025_clean.csv`).

Required columns used by the pipeline:

- `date` (YYYY-MM-DD)
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`
- `league` (SE1 filtering)
- `season` (for current table extraction)
- `competition` (must be `league` for seasonal baseline)

## Upcoming fixtures CSV

Configured by `fixtures_path` (default: `data/01_raw/upcoming_fixtures.csv`).

Required columns:

- `date` (YYYY-MM-DD)
- `home_team`
- `away_team`
- `league`
- `season`
- `competition_format` (`league` or `knockout`)
- `round` (integer)

Validation rules:

- Must contain all required columns.
- Must include rows for (`target_league`, `target_season`, `competition_format=league`).
- `round` must be non-null integers.
- Duplicate fixtures (`date`, `home_team`, `away_team`, `round`) are rejected.

# How to run it

## 1) Create and activate environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install dependencies

```bash
pip install -e .
```

## 3) Install CmdStan once (required by CmdStanPy)

```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

## 4) Run the full Kedro pipeline

```bash
kedro run
```

## 5) Check outputs

- `data/07_model_output/coming_predictions.json`
- `data/07_model_output/seasonal_predictions.json`
- `docs/coming_predictions.json`
- `docs/seasonal_predictions.json`

## Runtime parameters

Defaults are in `conf/base/parameters.yml`:

- `matches_path`
- `fixtures_path`
- `target_season` (current: 2026)
- `target_league` (current: SE1)
- `n_simulations` (current: 5000)
- `random_seed` (current: 42)
- `relegation_spots` (current: 3)

You can change them in config before running.

## Reproducibility notes

- Match prediction and season simulation are stochastic, but `random_seed` fixes simulation reproducibility.
- Stan sampling still has Monte Carlo uncertainty; increase iterations/chains for tighter posterior estimates.

## Known assumptions and limitations

- Only `SE1` is used in the default pipeline.
- Unknown teams in coming fixtures are allowed (flagged), but seasonal simulation requires all teams mapped.
- Scoreline table is truncated to 6 goals per side for exported probabilities.
- Tie-break implementation is generic and may not match every official competition edge case.
