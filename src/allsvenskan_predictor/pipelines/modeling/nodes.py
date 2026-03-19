import polars as pl
import numpy as np
import json
import shutil
from pathlib import Path

from cmdstanpy import CmdStanModel
from scipy.stats import skellam, poisson


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_matches(filepath) -> pl.DataFrame:

    df = pl.read_csv(filepath)

    df = df.with_columns(
        pl.col("date").str.to_date()
    )

    return df


def filter_se1_matches(df) -> pl.DataFrame:

    return df.filter(pl.col("league") == "SE1")


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def add_time_decay(df, xi=0.3) -> pl.DataFrame:

    latest_date = df.select(pl.col("date").max()).item()

    df = df.with_columns(
        ((latest_date - pl.col("date")).dt.total_days() / 365)
        .alias("age_years")
    )

    df = df.with_columns(
        pl.col("age_years")
        .map_elements(lambda x: np.exp(-xi * x))
        .alias("weight")
    )

    return df


# ---------------------------------------------------------
# TEAM MAPPING
# ---------------------------------------------------------

def create_team_mapping(df):

    teams = (
        pl.concat([
            df.select(pl.col("home_team").alias("team")),
            df.select(pl.col("away_team").alias("team"))
        ])
        .unique()
        .sort("team")
        .to_series()
        .to_list()
    )

    team_to_id = {team: i + 1 for i, team in enumerate(teams)}

    df = df.with_columns([

        pl.col("home_team")
        .map_elements(lambda x: team_to_id[x], return_dtype=pl.Int32)
        .alias("home_id"),

        pl.col("away_team")
        .map_elements(lambda x: team_to_id[x], return_dtype=pl.Int32)
        .alias("away_id")

    ])

    return df, team_to_id


# ---------------------------------------------------------
# STAN DATA
# ---------------------------------------------------------

def build_stan_data(df, team_to_id):

    stan_data = {

        "N": df.height,
        "T": len(team_to_id),

        "home_team": df["home_id"].to_numpy(),
        "away_team": df["away_id"].to_numpy(),

        "home_goals": df["home_goals"].to_numpy(),
        "away_goals": df["away_goals"].to_numpy(),

        "weights": df["weight"].to_numpy()
    }

    return stan_data


# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------

def train_model(stan_data):

    model = CmdStanModel(
        stan_file="src/allsvenskan_predictor/stan/poisson_model.stan"
    )

    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        adapt_delta=0.9,
        seed=42
    )

    return fit


# ---------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------

def load_fixtures(filepath):

    df = pl.read_csv(filepath)

    df = df.with_columns(
        pl.col("date").str.to_date()
    )

    return df


def map_fixture_teams(fixtures, team_map):

    # Optional: log missing teams (but don't crash)
    missing_home = set(fixtures["home_team"]) - set(team_map.keys())
    missing_away = set(fixtures["away_team"]) - set(team_map.keys())
    missing = missing_home.union(missing_away)

    if missing:
        print(f"[WARNING] Unknown teams encountered: {missing}")

    fixtures = fixtures.with_columns([

        pl.col("home_team")
        .map_elements(lambda x: team_map.get(x), return_dtype=pl.Int32)
        .alias("home_id"),

        pl.col("away_team")
        .map_elements(lambda x: team_map.get(x), return_dtype=pl.Int32)
        .alias("away_id")

    ])

    return fixtures


# ---------------------------------------------------------
# SCORELINE DISTRIBUTION
# ---------------------------------------------------------

def scoreline_table(lambda_home, lambda_away, max_goals=6):

    table = {}

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):

            p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)

            table[f"{h}-{a}"] = float(p)

    return table


# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

def generate_predictions(fit, fixtures):

    attack = fit.stan_variable("attack")
    defense = fit.stan_variable("defense")
    gamma = fit.stan_variable("gamma")

    draws = gamma.shape[0]

    predictions = []

    for row in fixtures.iter_rows(named=True):

        h_id = row["home_id"]
        a_id = row["away_id"]

        unknown_home = h_id is None
        unknown_away = a_id is None

        unknown_team = unknown_home or unknown_away

        # ---- Handle UNKNOWN TEAMS ----
        if unknown_home:
            attack_h = np.zeros(draws)
            defense_h = np.zeros(draws)
        else:
            h = h_id - 1
            attack_h = attack[:, h]
            defense_h = defense[:, h]

        if unknown_away:
            attack_a = np.zeros(draws)
            defense_a = np.zeros(draws)
        else:
            a = a_id - 1
            attack_a = attack[:, a]
            defense_a = defense[:, a]

        # ---- Expected goals ----
        lambda_home = np.exp(attack_h - defense_a + gamma)
        lambda_away = np.exp(attack_a - defense_h)

        lambda_home_mean = float(lambda_home.mean())
        lambda_away_mean = float(lambda_away.mean())

        # ---- Outcome probabilities ----
        p_home = 1 - skellam.cdf(0, lambda_home, lambda_away)
        p_draw = skellam.pmf(0, lambda_home, lambda_away)
        p_away = skellam.cdf(-1, lambda_home, lambda_away)

        p_home_mean = float(p_home.mean())
        p_draw_mean = float(p_draw.mean())
        p_away_mean = float(p_away.mean())

        # ---- Scoreline probabilities ----
        max_goals = 6
        score_probs = {}

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):

                p_ij = (
                    poisson.pmf(i, lambda_home)
                    * poisson.pmf(j, lambda_away)
                )

                score_probs[f"{i}-{j}"] = float(p_ij.mean())

        # ---- Base output ----
        output = {
            "date": str(row["date"]),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "lambda_home": lambda_home_mean,
            "lambda_away": lambda_away_mean,
            "scoreline_probs": score_probs,
            "unknown_team": unknown_team,
            "unknown_home": unknown_home,
            "unknown_away": unknown_away,
        }

        # ---- Competition logic ----
        if row.get("competition_format") == "knockout":

            p_home_adv = p_home_mean + 0.5 * p_draw_mean
            p_away_adv = p_away_mean + 0.5 * p_draw_mean

            output.update({
                "competition_format": "knockout",
                "p_home_win": p_home_mean,
                "p_draw": p_draw_mean,
                "p_away_win": p_away_mean,
                "p_home_advance": float(p_home_adv),
                "p_away_advance": float(p_away_adv),
            })

        else:
            output.update({
                "competition_format": "league",
                "p_home_win": p_home_mean,
                "p_draw": p_draw_mean,
                "p_away_win": p_away_mean,
            })

        predictions.append(output)

    return predictions


# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------

def export_artifacts(predictions):

    with open("data/07_model_output/predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    return True


def publish_to_docs(predictions):

    output_dir = Path("data/07_model_output")
    docs_dir = Path("docs")

    docs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Source files ----
    pred_src = output_dir / "predictions.json"

    # ---- Destination ----
    pred_dst = docs_dir / "predictions.json"

    # ---- Copy ----
    shutil.copy(pred_src, pred_dst)

    return None