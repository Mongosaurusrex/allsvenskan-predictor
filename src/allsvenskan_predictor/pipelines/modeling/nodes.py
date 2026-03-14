import polars as pl
import numpy as np
import json

from cmdstanpy import CmdStanModel
from scipy.stats import skellam


def load_matches(filepath) -> pl.DataFrame:

    df = pl.read_csv(filepath)

    df = df.with_columns(
        pl.col("date").str.to_date()
    )

    return df


def filter_se1_matches(df) -> pl.DataFrame:

    return df.filter(pl.col("league") == "SE1")


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


def load_fixtures(filepath):

    df = pl.read_csv(filepath)

    df = df.with_columns(
        pl.col("date").str.to_date()
    )

    return df


def map_fixture_teams(fixtures, team_map):

    missing_home = set(fixtures["home_team"]) - set(team_map.keys())
    missing_away = set(fixtures["away_team"]) - set(team_map.keys())

    missing = missing_home.union(missing_away)

    if missing:
        raise ValueError(
            f"Unknown teams in fixtures not present in training data: {missing}"
        )

    fixtures = fixtures.with_columns([

        pl.col("home_team")
        .map_elements(lambda x: team_map[x], return_dtype=pl.Int32)
        .alias("home_id"),

        pl.col("away_team")
        .map_elements(lambda x: team_map[x], return_dtype=pl.Int32)
        .alias("away_id")

    ])

    return fixtures


def generate_predictions(fit, fixtures):

    attack = fit.stan_variable("attack")
    defense = fit.stan_variable("defense")
    gamma = fit.stan_variable("gamma")

    predictions = []

    for row in fixtures.iter_rows(named=True):

        h = row["home_id"] - 1
        a = row["away_id"] - 1

        lambda_home = np.exp(
            attack[:, h] - defense[:, a] + gamma
        )

        lambda_away = np.exp(
            attack[:, a] - defense[:, h]
        )

        p_home = 1 - skellam.cdf(0, lambda_home, lambda_away)
        p_draw = skellam.pmf(0, lambda_home, lambda_away)
        p_away = skellam.cdf(-1, lambda_home, lambda_away)

        predictions.append({

            "date": str(row["date"]),
            "home_team": row["home_team"],
            "away_team": row["away_team"],

            "lambda_home": float(lambda_home.mean()),
            "lambda_away": float(lambda_away.mean()),

            "p_home": float(p_home.mean()),
            "p_draw": float(p_draw.mean()),
            "p_away": float(p_away.mean())
        })

    return predictions


def export_artifacts(predictions):

    with open("data/07_model_output/predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    return True
