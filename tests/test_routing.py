from agents.routing import (
    extract_research_command,
    should_create_report,
    should_search_web,
    should_use_sandbox,
    should_use_statistical_simulation,
)


def test_trivial_questions_use_concise_chat_answer_path() -> None:
    assert not should_create_report("What is 2 + 2?")
    assert not should_create_report("What is the distance between Pune and Hyderabad?")


def test_explicit_and_complex_requests_use_full_research() -> None:
    assert should_create_report("Write a research report about battery recycling.")
    assert should_create_report(
        "Compare these three approaches across cost, reliability, environmental impact, and long-term maintenance."
    )
    question, forced = extract_research_command("/research Compare the top battery storage approaches")
    assert forced
    assert question == "Compare the top battery storage approaches"


def test_code_requests_route_to_the_sandbox_tool() -> None:
    assert should_use_sandbox("Please run Python to simulate these outcomes")
    assert not should_use_sandbox("What is Python used for?")


def test_statistical_modeling_routes_to_the_in_chat_simulation_tool() -> None:
    assert should_use_statistical_simulation("Use Monte Carlo simulation to estimate portfolio risk")
    assert should_use_statistical_simulation("Build a statistical model to forecast the next year")
    assert should_use_statistical_simulation("What is the probability this investment drops 30% given its volatility?")
    assert not should_use_statistical_simulation("What is Monte Carlo simulation?")
    assert not should_use_statistical_simulation("Explain statistical modeling")


def test_trivial_chat_does_not_trigger_web_or_document_retrieval() -> None:
    assert not should_search_web("What is 2 + 2?")
    assert not should_search_web("Thanks!")
    assert should_search_web("What is the distance between Pune and Hyderabad?")
