from kedro.pipeline import Pipeline, node

from .nodes import (
    load_matches,
    filter_se1_matches,
    add_time_decay,
    create_team_mapping,
    build_stan_data,
    publish_to_docs,
    train_model,
    load_fixtures,
    validate_fixtures,
    map_fixture_teams,
    select_coming_round_fixtures,
    generate_predictions,
    generate_seasonal_predictions,
    export_coming_predictions,
    export_seasonal_predictions,
)


def create_pipeline() -> Pipeline:

    return Pipeline([

        node(
            load_matches,
            inputs="params:matches_path",
            outputs="matches",
        ),

        node(
            filter_se1_matches,
            inputs="matches",
            outputs="se1_matches",
        ),

        node(
            add_time_decay,
            inputs="se1_matches",
            outputs="weighted_matches",
        ),

        node(
            create_team_mapping,
            inputs="weighted_matches",
            outputs=["mapped_matches", "team_map"],
        ),

        node(
            build_stan_data,
            inputs=["mapped_matches", "team_map"],
            outputs="stan_data",
        ),

        node(
            train_model,
            inputs="stan_data",
            outputs="model_fit",
        ),

        node(
            load_fixtures,
            inputs="params:fixtures_path",
            outputs="fixtures_raw",
        ),

        node(
            validate_fixtures,
            inputs=["fixtures_raw", "params:target_season", "params:target_league"],
            outputs="fixtures",
        ),

        node(
            map_fixture_teams,
            inputs=["fixtures", "team_map"],
            outputs="mapped_fixtures",
        ),

        node(
            select_coming_round_fixtures,
            inputs="mapped_fixtures",
            outputs="coming_fixtures",
        ),

        node(
            generate_predictions,
            inputs=["model_fit", "coming_fixtures"],
            outputs="coming_predictions",
        ),

        node(
            generate_seasonal_predictions,
            inputs=[
                "model_fit",
                "mapped_fixtures",
                "matches",
                "params:target_season",
                "params:target_league",
                "params:n_simulations",
                "params:random_seed",
                "params:relegation_spots",
            ],
            outputs="seasonal_predictions",
        ),

        node(
            export_coming_predictions,
            inputs="coming_predictions",
            outputs="coming_artifacts_written",
        ),

        node(
            export_seasonal_predictions,
            inputs="seasonal_predictions",
            outputs="seasonal_artifacts_written",
        ),

        node(
            publish_to_docs,
            inputs=["coming_predictions", "seasonal_predictions"],
            outputs=None,
        )
    ])
