import polars as pl
import numpy as np
import json
import shutil
from pathlib import Path

from cmdstanpy import CmdStanModel
from scipy.stats import skellam, poisson


REQUIRED_FIXTURE_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "league",
    "season",
    "competition_format",
    "round",
}


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

def add_time_decay(df, xi=0.7) -> pl.DataFrame:

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


def validate_fixtures(fixtures, target_season, target_league):

    missing_columns = REQUIRED_FIXTURE_COLUMNS - set(fixtures.columns)
    if missing_columns:
        raise ValueError(
            "Missing required fixture columns: "
            f"{sorted(missing_columns)}"
        )

    filtered = fixtures.filter(
        (pl.col("season") == target_season)
        & (pl.col("league") == target_league)
        & (pl.col("competition_format") == "league")
    )

    if filtered.is_empty():
        raise ValueError(
            f"No fixtures found for league={target_league}, season={target_season}."
        )

    filtered = filtered.with_columns(pl.col("round").cast(pl.Int32, strict=True))

    if filtered.filter(pl.col("round").is_null()).height > 0:
        raise ValueError("Fixture round column contains null values.")

    duplicate_rows = filtered.select([
        "date",
        "home_team",
        "away_team",
        "round",
    ]).is_duplicated()

    if filtered.filter(duplicate_rows).height > 0:
        raise ValueError("Duplicate fixtures found in upcoming fixtures input.")

    return filtered.sort(["round", "date", "home_team"])


def select_coming_round_fixtures(fixtures):

    min_round = fixtures.select(pl.col("round").min()).item()

    return fixtures.filter(pl.col("round") == min_round).sort([
        "date",
        "home_team",
        "away_team",
    ])


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


def _empty_standing_row():

    return {
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
        "points": 0,
    }


def _apply_match_result(standings, home_team, away_team, home_goals, away_goals):

    home = standings[home_team]
    away = standings[away_team]

    home["played"] += 1
    away["played"] += 1

    home["goals_for"] += int(home_goals)
    home["goals_against"] += int(away_goals)
    away["goals_for"] += int(away_goals)
    away["goals_against"] += int(home_goals)

    home["goal_diff"] = home["goals_for"] - home["goals_against"]
    away["goal_diff"] = away["goals_for"] - away["goals_against"]

    if home_goals > away_goals:
        home["wins"] += 1
        away["losses"] += 1
        home["points"] += 3
    elif home_goals < away_goals:
        away["wins"] += 1
        home["losses"] += 1
        away["points"] += 3
    else:
        home["draws"] += 1
        away["draws"] += 1
        home["points"] += 1
        away["points"] += 1


def _rank_teams(standings):

    return sorted(
        standings,
        key=lambda team: (
            -standings[team]["points"],
            -standings[team]["goal_diff"],
            -standings[team]["goals_for"],
            team,
        ),
    )


def generate_seasonal_predictions(
    fit,
    fixtures,
    matches,
    target_season,
    target_league,
    n_simulations,
    random_seed,
    relegation_spots,
):

    unknown = fixtures.filter(
        pl.col("home_id").is_null() | pl.col("away_id").is_null()
    )
    if unknown.height > 0:
        raise ValueError("Seasonal simulation requires known team mappings for all fixtures.")

    current_matches = matches.filter(
        (pl.col("season") == target_season)
        & (pl.col("league") == target_league)
        & (pl.col("competition") == "league")
    )

    teams = sorted(
        set(fixtures["home_team"]).union(set(fixtures["away_team"]))
    )

    base_standings = {team: _empty_standing_row() for team in teams}
    teams_set = set(teams)

    for row in current_matches.iter_rows(named=True):
        home_team = row["home_team"]
        away_team = row["away_team"]
        
        if home_team in teams_set and away_team in teams_set:
            _apply_match_result(
                base_standings,
                home_team,
                away_team,
                row["home_goals"],
                row["away_goals"],
            )

    fixtures_sorted = fixtures.sort(["round", "date", "home_team", "away_team"])

    attack = fit.stan_variable("attack")
    defense = fit.stan_variable("defense")
    gamma = fit.stan_variable("gamma")
    draws = gamma.shape[0]

    rng = np.random.default_rng(random_seed)
    n_teams = len(teams)
    position_counts = {team: np.zeros(n_teams, dtype=np.int32) for team in teams}
    points_samples = {team: [] for team in teams}

    for _ in range(n_simulations):
        standings = {team: stats.copy() for team, stats in base_standings.items()}

        for row in fixtures_sorted.iter_rows(named=True):
            draw_idx = rng.integers(0, draws)

            home_idx = row["home_id"] - 1
            away_idx = row["away_id"] - 1

            lambda_home = np.exp(
                attack[draw_idx, home_idx] - defense[draw_idx, away_idx] + gamma[draw_idx]
            )
            lambda_away = np.exp(
                attack[draw_idx, away_idx] - defense[draw_idx, home_idx]
            )

            home_goals = int(rng.poisson(lambda_home))
            away_goals = int(rng.poisson(lambda_away))

            _apply_match_result(
                standings,
                row["home_team"],
                row["away_team"],
                home_goals,
                away_goals,
            )

        ranking = _rank_teams(standings)

        for pos, team in enumerate(ranking, start=1):
            position_counts[team][pos - 1] += 1

        for team in teams:
            points_samples[team].append(standings[team]["points"])

    team_summaries = []
    relegation_spots = min(relegation_spots, n_teams)

    for team in teams:
        probs = position_counts[team] / n_simulations
        top_n = min(3, n_teams)

        team_summary = {
            "team": team,
            "position_probs": {
                str(pos + 1): float(prob) for pos, prob in enumerate(probs)
            },
            "p_champion": float(probs[0]),
            "p_top3": float(probs[:top_n].sum()),
            "p_relegation": float(probs[-relegation_spots:].sum()),
            "expected_points": float(np.mean(points_samples[team])),
            "points_p10": float(np.quantile(points_samples[team], 0.1)),
            "points_p90": float(np.quantile(points_samples[team], 0.9)),
        }
        team_summaries.append(team_summary)

    team_summaries.sort(
        key=lambda row: (
            -row["p_champion"],
            -row["expected_points"],
            row["team"],
        )
    )

    min_round = int(fixtures_sorted.select(pl.col("round").min()).item())
    max_round = int(fixtures_sorted.select(pl.col("round").max()).item())

    return {
        "league": target_league,
        "season": target_season,
        "simulations": int(n_simulations),
        "random_seed": int(random_seed),
        "rounds_completed": min_round - 1,
        "min_upcoming_round": min_round,
        "max_upcoming_round": max_round,
        "total_remaining_rounds": max_round - min_round + 1,
        "teams": team_summaries,
    }


# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------

def export_coming_predictions(predictions):

    with open("data/07_model_output/coming_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    return True


def export_seasonal_predictions(predictions):

    with open("data/07_model_output/seasonal_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    return True


def publish_to_docs(coming_predictions, seasonal_predictions):

    del coming_predictions
    del seasonal_predictions

    output_dir = Path("data/07_model_output")
    docs_dir = Path("docs")

    docs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Source files ----
    coming_src = output_dir / "coming_predictions.json"
    seasonal_src = output_dir / "seasonal_predictions.json"

    # ---- Destination ----
    coming_dst = docs_dir / "coming_predictions.json"
    seasonal_dst = docs_dir / "seasonal_predictions.json"

    # ---- Copy ----
    shutil.copy(coming_src, coming_dst)
    shutil.copy(seasonal_src, seasonal_dst)

    return None