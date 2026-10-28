from __future__ import annotations

import pytest

from guardrails import GuardrailViolation
from providers.base import GenerationResponse
from tools.statistical_simulation import (
    StatisticalSimulationTool,
    parse_simulation_plan,
    run_simulation,
    simulation_plan_hash,
)


def test_normal_threshold_simulation_is_reproducible_and_accurate() -> None:
    plan = parse_simulation_plan({
        "description": "Estimate the probability of a loss greater than 20%.",
        "method": "normal_threshold_probability",
        "parameters": {
            "mean": 0.07,
            "standard_deviation": 0.18,
            "threshold": -0.20,
            "iterations": 100_000,
            "seed": 7,
        },
    })

    result = run_simulation(plan, approved=True)
    repeated = run_simulation(plan, approved=True)

    assert result.status == "completed"
    assert result.metrics["estimated_probability"] == pytest.approx(0.0668, abs=0.004)
    assert result.metrics["estimated_probability"] == repeated.metrics["estimated_probability"]
    assert "Monte Carlo interval" in result.summary


def test_binomial_and_bootstrap_methods_return_bounded_metrics() -> None:
    binomial = parse_simulation_plan({
        "description": "Estimate the chance of at least 5 successes.",
        "method": "binomial_tail_probability",
        "parameters": {
            "success_probability": 0.5,
            "trials_per_experiment": 10,
            "minimum_successes": 5,
            "iterations": 20_000,
            "seed": 3,
        },
    })
    bootstrap = parse_simulation_plan({
        "description": "Bootstrap a confidence interval for the sample mean.",
        "method": "bootstrap_mean_interval",
        "parameters": {
            "values": [1, 2, 3, 4, 5],
            "iterations": 5000,
            "confidence_level": 0.95,
            "seed": 11,
        },
    })

    assert run_simulation(binomial, approved=True).metrics["estimated_probability"] == pytest.approx(0.623, abs=0.02)
    bootstrap_result = run_simulation(bootstrap, approved=True)
    low, high = bootstrap_result.metrics["confidence_interval"]
    assert low <= bootstrap_result.metrics["sample_mean"] <= high


def test_simulation_plans_reject_unapproved_code_and_unbounded_parameters() -> None:
    with pytest.raises(GuardrailViolation, match="approve"):
        run_simulation(parse_simulation_plan({
            "description": "A fixed plan.",
            "method": "normal_threshold_probability",
            "parameters": {
                "mean": 0.0, "standard_deviation": 1.0, "threshold": 0.0,
                "iterations": 100, "seed": 1,
            },
        }), approved=False)
    with pytest.raises(GuardrailViolation, match="method"):
        parse_simulation_plan({
            "description": "Run arbitrary code.",
            "method": "python",
            "parameters": {"code": "import os; print(os.listdir())"},
        })
    with pytest.raises(GuardrailViolation, match="iterations"):
        parse_simulation_plan({
            "description": "Too many simulations.",
            "method": "normal_threshold_probability",
            "parameters": {
                "mean": 0.0, "standard_deviation": 1.0, "threshold": 0.0,
                "iterations": 1_000_000_000, "seed": 1,
            },
        })


def test_simulation_tool_requests_a_validated_plan_without_source_code() -> None:
    class Provider:
        def __init__(self):
            self.prompt = ""

        def generate_text(self, model_id, prompt, **kwargs):
            self.prompt = prompt
            return GenerationResponse(
                '{"description":"Estimate a threshold probability.",'
                '"method":"normal_threshold_probability",'
                '"parameters":{"mean":0.07,"standard_deviation":0.18,"threshold":-0.2,"iterations":10000,"seed":4}}',
                model_id,
            )

    provider = Provider()
    plan = StatisticalSimulationTool().build(provider, "test-model", "Monte Carlo portfolio loss estimate")

    assert plan.method == "normal_threshold_probability"
    assert "Do not return source code" in provider.prompt
    assert len(simulation_plan_hash(plan)) == 64
