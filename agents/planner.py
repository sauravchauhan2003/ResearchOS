from __future__ import annotations

from agents.base_agent import BaseAgent, parse_json_object
from agents.models import ResearchTask


class Planner(BaseAgent):
    def plan(self, question: str, *, objectives: str = "", max_agents: int = 2) -> list[ResearchTask]:
        limit = max(1, min(int(max_agents), 3))
        prompt = (
            "Create a concise research plan for the question below. Return JSON only as "
            '{"subquestions":["..."]}. Make independent questions that can be researched in parallel. '
            f"Return between 1 and {limit} subquestions. Do not answer the research question.\n\n"
            f"Question: {question}\nObjectives and constraints: {objectives or 'None supplied'}"
        )
        try:
            raw = self.complete(prompt, system_instruction="You are the lead researcher and planner.", max_output_tokens=600)
            parsed = parse_json_object(raw)
            questions = parsed.get("subquestions")
            if not isinstance(questions, list):
                raise ValueError("Missing subquestions list.")
            clean = list(dict.fromkeys(" ".join(str(item).split()) for item in questions if str(item).strip()))[:limit]
            if not clean:
                raise ValueError("No usable subquestions.")
        except Exception:
            clean = [" ".join(part.split()) for part in objectives.splitlines() if part.strip()][:limit]
            if not clean:
                clean = [question.strip()]
        return [ResearchTask(f"research-{index + 1}", item) for index, item in enumerate(clean)]
