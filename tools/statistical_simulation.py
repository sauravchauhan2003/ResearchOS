from __future__ import annotations

import json
import hashlib
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from guardrails import GuardrailViolation
from observability import log_event


MAX_ITERATIONS = 250_000
MAX_BOOTSTRAP_CELLS = 3_000_000
MAX_BOOTSTRAP_VALUES = 500
MAX_PARAMETER_MAGNITUDE = 1_000_000.0
METHODS = {
    "normal_threshold_probability",
    "binomial_tail_probability",
    "bootstrap_mean_interval",
}
_SIMULATION_SLOTS = threading.BoundedSemaphore(2)


@dataclass(frozen=True, slots=True)
class SimulationPlan:
    description: str
    method: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    run_id: str
    status: str
    method: str
    summary: str
    metrics: dict[str, Any]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any, name: str, *, minimum: float = -MAX_PARAMETER_MAGNITUDE,
            maximum: float = MAX_PARAMETER_MAGNITUDE) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardrailViolation(f"Simulation parameter '{name}' must be a number.")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise GuardrailViolation(f"Simulation parameter '{name}' is outside the allowed range.")
    return number


def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise GuardrailViolation(f"Simulation parameter '{name}' must be an integer from {minimum} to {maximum}.")
    return value


def parse_simulation_plan(payload: dict[str, Any]) -> SimulationPlan:
    """Validate a model-produced JSON plan against the fixed, non-code simulation API."""
    if not isinstance(payload, dict) or set(payload) != {"description", "method", "parameters"}:
        raise GuardrailViolation("The model must return a description, method, and parameters object.")
    description = payload["description"]
    method = payload["method"]
    raw = payload["parameters"]
    if not isinstance(description, str) or not description.strip() or len(description) > 500:
        raise GuardrailViolation("The simulation description is missing or too long.")
    if not isinstance(method, str) or method not in METHODS or not isinstance(raw, dict):
        raise GuardrailViolation("The requested statistical method is not supported.")

    if method == "normal_threshold_probability":
        required = {"mean", "standard_deviation", "threshold", "iterations", "seed"}
        if set(raw) != required:
            raise GuardrailViolation("Normal threshold simulations require mean, standard deviation, threshold, iterations, and seed.")
        parameters = {
            "mean": _number(raw["mean"], "mean"),
            "standard_deviation": _number(raw["standard_deviation"], "standard_deviation", minimum=0.000000001),
            "threshold": _number(raw["threshold"], "threshold"),
            "iterations": _integer(raw["iterations"], "iterations", minimum=100, maximum=MAX_ITERATIONS),
            "seed": _integer(raw["seed"], "seed", minimum=0, maximum=4_294_967_295),
        }
    elif method == "binomial_tail_probability":
        required = {"success_probability", "trials_per_experiment", "minimum_successes", "iterations", "seed"}
        if set(raw) != required:
            raise GuardrailViolation("Binomial simulations require success_probability, trials_per_experiment, minimum_successes, iterations, and seed.")
        parameters = {
            "success_probability": _number(raw["success_probability"], "success_probability", minimum=0, maximum=1),
            "trials_per_experiment": _integer(raw["trials_per_experiment"], "trials_per_experiment", minimum=1, maximum=100_000),
            "minimum_successes": _integer(raw["minimum_successes"], "minimum_successes", minimum=0, maximum=100_000),
            "iterations": _integer(raw["iterations"], "iterations", minimum=100, maximum=MAX_ITERATIONS),
            "seed": _integer(raw["seed"], "seed", minimum=0, maximum=4_294_967_295),
        }
        if parameters["minimum_successes"] > parameters["trials_per_experiment"]:
            raise GuardrailViolation("minimum_successes cannot exceed trials_per_experiment.")
    else:
        required = {"values", "iterations", "confidence_level", "seed"}
        if set(raw) != required:
            raise GuardrailViolation("Bootstrap intervals require values, iterations, confidence_level, and seed.")
        values = raw["values"]
        if not isinstance(values, list) or not 2 <= len(values) <= MAX_BOOTSTRAP_VALUES:
            raise GuardrailViolation(f"Bootstrap data must contain 2 to {MAX_BOOTSTRAP_VALUES} numeric values.")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                   and abs(value) <= MAX_PARAMETER_MAGNITUDE for value in values):
            raise GuardrailViolation("Bootstrap values must be finite, bounded numbers.")
        iterations = _integer(raw["iterations"], "iterations", minimum=100,
                              maximum=min(MAX_ITERATIONS, MAX_BOOTSTRAP_CELLS // len(values)))
        parameters = {
            "values": [float(value) for value in values],
            "iterations": iterations,
            "confidence_level": _number(raw["confidence_level"], "confidence_level", minimum=0.5, maximum=0.999),
            "seed": _integer(raw["seed"], "seed", minimum=0, maximum=4_294_967_295),
        }
    return SimulationPlan(description.strip(), method, parameters)


def simulation_plan_hash(plan: SimulationPlan | dict[str, Any]) -> str:
    payload = plan.to_dict() if isinstance(plan, SimulationPlan) else plan
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def run_simulation(plan: SimulationPlan, *, approved: bool) -> SimulationResult:
    if not approved:
        raise GuardrailViolation("Review and approve this exact simulation plan before running it.")
    plan = parse_simulation_plan(plan.to_dict())
    parameters = plan.parameters
    rng = np.random.default_rng(parameters["seed"])
    started = time.perf_counter()
    metrics: dict[str, Any]

    if plan.method == "normal_threshold_probability":
        draws = rng.normal(parameters["mean"], parameters["standard_deviation"], parameters["iterations"])
        probability = float(np.mean(draws <= parameters["threshold"]))
        standard_error = math.sqrt(probability * (1.0 - probability) / parameters["iterations"])
        interval = [max(0.0, probability - 1.96 * standard_error),
                    min(1.0, probability + 1.96 * standard_error)]
        metrics = {
            "estimated_probability": probability,
            "standard_error": standard_error,
            "confidence_interval_95": interval,
            "mean": parameters["mean"],
            "standard_deviation": parameters["standard_deviation"],
            "threshold": parameters["threshold"],
            "iterations": parameters["iterations"],
            "seed": parameters["seed"],
        }
        summary = (
            f"Estimated probability of being at or below {parameters['threshold']:.6g}: {probability:.2%} "
            f"(approximate 95% Monte Carlo interval {interval[0]:.2%}–{interval[1]:.2%}; "
            f"{parameters['iterations']:,} draws)."
        )
    elif plan.method == "binomial_tail_probability":
        draws = rng.binomial(parameters["trials_per_experiment"], parameters["success_probability"],
                             parameters["iterations"])
        probability = float(np.mean(draws >= parameters["minimum_successes"]))
        standard_error = math.sqrt(probability * (1.0 - probability) / parameters["iterations"])
        interval = [max(0.0, probability - 1.96 * standard_error),
                    min(1.0, probability + 1.96 * standard_error)]
        metrics = {
            "estimated_probability": probability,
            "standard_error": standard_error,
            "confidence_interval_95": interval,
            "success_probability": parameters["success_probability"],
            "trials_per_experiment": parameters["trials_per_experiment"],
            "minimum_successes": parameters["minimum_successes"],
            "iterations": parameters["iterations"],
            "seed": parameters["seed"],
        }
        summary = (
            f"Estimated probability of at least {parameters['minimum_successes']} successes in "
            f"{parameters['trials_per_experiment']} trials: {probability:.2%} "
            f"(approximate 95% Monte Carlo interval {interval[0]:.2%}–{interval[1]:.2%}; "
            f"{parameters['iterations']:,} simulations)."
        )
    else:
        values = np.asarray(parameters["values"], dtype=np.float64)
        draws = rng.integers(0, len(values), size=(parameters["iterations"], len(values)))
        bootstrap_means = values[draws].mean(axis=1)
        tail = (1.0 - parameters["confidence_level"]) / 2.0
        lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
        estimate = float(values.mean())
        interval = [float(lower), float(upper)]
        metrics = {
            "sample_mean": estimate,
            "confidence_interval": interval,
            "confidence_level": parameters["confidence_level"],
            "sample_size": len(values),
            "iterations": parameters["iterations"],
            "seed": parameters["seed"],
        }
        summary = (
            f"The sample mean is {estimate:.6g}; the bootstrap "
            f"{parameters['confidence_level']:.1%} interval is [{lower:.6g}, {upper:.6g}] "
            f"from {parameters['iterations']:,} resamples."
        )

    duration = time.perf_counter() - started
    log_event("statistical_simulation_finished", status="complete", tool=plan.method,
              duration_ms=round(duration * 1000), result_count=parameters["iterations"])
    return SimulationResult(str(uuid.uuid4()), "completed", plan.method, summary, metrics, round(duration, 6))


class StatisticalSimulationTool:
    """Build validated statistical plans and run only the supported local algorithms."""

    def build(self, provider, model_id: str, request: str) -> SimulationPlan:
        from agents.base_agent import parse_json_object
        from agents.context import compact_text

        prompt = (
            "Choose a supported statistical operation and return one JSON object with exactly these keys: "
            "description, method, parameters. Do not return source code. Supported methods and exact parameters: "
            "normal_threshold_probability(mean, standard_deviation, threshold, iterations, seed); "
            "binomial_tail_probability(success_probability, trials_per_experiment, minimum_successes, iterations, seed); "
            "bootstrap_mean_interval(values, iterations, confidence_level, seed). Numeric rates use decimals "
            "(7% is 0.07). Use at most 250000 iterations; bootstrap uses at most 500 values and the app "
            "also bounds total resamples. Choose a fixed integer seed for reproducibility. If essential data or "
            "assumptions are missing, explain what is missing in description and use a clearly stated assumption, "
            "never fabricate observed data. User request:\n" + compact_text(request, 1800)
        )
        response = provider.generate_text(
            model_id,
            prompt,
            system_instruction=(
                "You select parameters for a fixed statistical simulation tool. Your JSON is untrusted input and "
                "will be checked against a strict schema. Do not generate code or request secrets. State assumptions."
            ),
            max_output_tokens=900,
            temperature=0.1,
        )
        return parse_simulation_plan(parse_json_object(response.text))

    def run(self, plan: SimulationPlan | dict[str, Any], *, approved: bool) -> SimulationResult:
        validated = parse_simulation_plan(plan.to_dict() if isinstance(plan, SimulationPlan) else plan)
        if not _SIMULATION_SLOTS.acquire(blocking=False):
            log_event("statistical_simulation_rejected", status="blocked", reason="capacity_limit")
            raise GuardrailViolation("Two simulations are already running. Wait briefly and try again.")
        try:
            return run_simulation(validated, approved=approved)
        finally:
            _SIMULATION_SLOTS.release()
