from __future__ import annotations

import re


REPORT_TERMS = (
    "research report", "write a report", "create a report", "comprehensive report",
    "deep research", "literature review", "white paper", "whitepaper", "detailed analysis",
)
COMPLEX_TERMS = (
    "compare", "comparison", "pros and cons", "trade-offs", "tradeoffs", "evaluate",
    "recommendation", "rank", "across multiple", "systematic", "impact of", "root cause",
)
SIMULATION_EXPLICIT_TERMS = (
    "simulate", "monte carlo", "bootstrap", "stochastic simulation", "statistical simulation",
    "run python", "run this code", "execute code", "test this code", "debug this code",
    "sandbox", "write and run", "calculate using code",
)
STATISTICAL_MODEL_TERMS = (
    "statistical model", "statistical modelling", "statistical modeling", "probabilistic model",
    "uncertainty model", "sampling distribution", "scenario model", "probability", "chance",
    "confidence interval", "distribution", "random", "uncertainty", "forecast", "risk", "volatility",
)
MODELING_ACTION_TERMS = (
    "estimate", "forecast", "predict", "project", "model", "compare", "simulate", "simulation",
    "run", "calculate", "what if", "what-if", "what is the probability", "how likely", "what are the odds",
)
EXPLANATION_PREFIXES = ("what is ", "what are ", "explain ", "define ", "how does ", "how do ")
MODELING_VERBS = ("simulate", "run", "execute", "build", "estimate", "forecast", "predict", "project", "calculate", "compare")


def extract_research_command(text: str) -> tuple[str, bool]:
    """Return the prompt with an optional /research override removed."""
    clean = text.strip()
    prefix = "/research"
    if clean.casefold() == prefix:
        return "", True
    if clean.casefold().startswith(prefix + " "):
        return clean[len(prefix):].strip(), True
    return clean, False


def should_create_report(question: str, *, force_report: bool = False) -> bool:
    """Use the full workflow only for explicit or clearly multi-part investigations."""
    normalized = " ".join(question.casefold().split())
    if force_report or any(term in normalized for term in REPORT_TERMS):
        return True
    if normalized.count("?") >= 2:
        return True
    words = re.findall(r"[\w'-]+", normalized)
    has_complex_signal = any(term in normalized for term in COMPLEX_TERMS)
    has_multiple_dimensions = normalized.count(",") >= 2 or normalized.count(" and ") >= 2
    if len(words) >= 12 and has_multiple_dimensions and has_complex_signal:
        return True
    return len(words) >= 20 and has_complex_signal


def should_use_statistical_simulation(question: str) -> bool:
    """Route explicit statistical simulation or code requests to the approval-gated tool."""
    normalized = " ".join(question.casefold().split())
    explicit = any(term in normalized for term in SIMULATION_EXPLICIT_TERMS)
    if explicit:
        if normalized.startswith(EXPLANATION_PREFIXES) and not any(verb in normalized for verb in MODELING_VERBS):
            return False
        return True
    has_statistical_context = any(term in normalized for term in STATISTICAL_MODEL_TERMS)
    has_modeling_action = any(term in normalized for term in MODELING_ACTION_TERMS)
    is_concept_question = (
        normalized.startswith(EXPLANATION_PREFIXES)
        and any(term in normalized for term in ("statistical model", "probabilistic model", "monte carlo", "bootstrap"))
        and not any(verb in normalized for verb in MODELING_VERBS)
    )
    if is_concept_question:
        return False
    return has_statistical_context and has_modeling_action


def should_use_sandbox(question: str) -> bool:
    """Backward-compatible alias for older sandbox-routing callers."""
    return should_use_statistical_simulation(question)


def should_search_web(question: str) -> bool:
    normalized = " ".join(question.casefold().split()).strip(" ?!.,")
    if not normalized:
        return False
    if normalized in {"hi", "hello", "hey", "thanks", "thank you", "who are you", "what can you do"}:
        return False
    arithmetic = re.sub(r"^(?:what is|calculate|compute)\s+", "", normalized)
    if re.fullmatch(r"[\d\s()+\-*/.%^=]+", arithmetic):
        return False
    return True
