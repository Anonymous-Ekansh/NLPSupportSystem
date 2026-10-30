"""End-to-end two-agent pipeline: query → A1 → A2 → result."""

from nlp.agent_a1 import IntentAnalyst
from nlp.agent_a2 import ResponseSpecialist


def run_pipeline(query: str, a1: IntentAnalyst, a2: ResponseSpecialist) -> dict:
    """Chain A1 and A2 on a single query."""
    a1_out = a1.analyze(query)
    a2_out = a2.respond(query, a1_out)
    return {"query": query, "a1_output": a1_out, "a2_output": a2_out}
