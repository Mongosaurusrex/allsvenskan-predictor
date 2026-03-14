from kedro.pipeline import Pipeline, node

from .nodes import (
    load_matches,
    filter_se1_matches,
    add_time_decay,
    create_team_mapping,
    build_stan_data,
    train_model,
    load_fixtures,
    map_fixture_teams,
    generate_predictions,
    export_artifacts,
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
            outputs="fixtures",
        ),

        node(
            map_fixture_teams,
            inputs=["fixtures", "team_map"],
            outputs="mapped_fixtures",
        ),

        node(
            generate_predictions,
            inputs=["model_fit", "mapped_fixtures"],
            outputs="predictions",
        ),

        node(
            export_artifacts,
            inputs="predictions",
            outputs="artifacts_written",
        ),
    ])
