"""Example configs stay loadable and recommend compatible scenarios."""

from pathlib import Path

import pytest

from ltm100.config import build_backend, build_dataset, load_config
from ltm100.core.scenarios import get_scenario

EXAMPLES = Path(__file__).parents[1] / "examples"

RECOMMENDED_SCENARIOS = {
    "memmachine.yaml": ("chat-replay",),
    "memmachine-mcp.yaml": ("chat-replay",),
    "synthetic.yaml": ("add-load", "search-load", "mixed"),
    "mem0.yaml": ("add-load", "search-load", "mixed"),
    "shared-project.yaml": ("add-load", "search-load", "mixed"),
}


@pytest.mark.parametrize("filename", RECOMMENDED_SCENARIOS)
def test_example_config_builds_adapters(filename: str) -> None:
    config = load_config(EXAMPLES / filename)

    assert build_dataset(config.dataset).name == config.dataset.name
    assert build_backend(config.backend).name == config.backend.name


@pytest.mark.parametrize(
    ("filename", "scenario_name"),
    [
        (filename, scenario)
        for filename, scenarios in RECOMMENDED_SCENARIOS.items()
        for scenario in scenarios
    ],
)
def test_example_recommends_a_compatible_scenario(
    filename: str, scenario_name: str
) -> None:
    config = load_config(EXAMPLES / filename)
    dataset = build_dataset(config.dataset)
    scenario = get_scenario(scenario_name)
    validate = getattr(scenario, "validate", None)

    if validate is not None:
        validate(dataset)
    else:
        user = dataset.users(1, seed=0)[0]
        assert next(scenario.plan(user, dataset, {"seed": 0})) is not None
