from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAuditAgent
from src.agents.schemas import AgentVerdict


COUNCIL_PROMPT = """Audit this software issue from three named perspectives in ONE model response.
These are perspectives from the same model, not independent agents.

Policy perspective: Decide whether the text describes an observed malfunction or a new feature/request.
Data perspective: Assess available metadata and graph evidence; missing fields are normal and are not negative evidence.
Pattern perspective: Look for concrete defect signatures (crash, exception, regression) versus enhancement language.
Do not let status, resolution, or priority alone determine the label. Evidence is suggestive, not proof.

Return only valid JSON with this exact shape:
{"policy":{"label":"bug|non-bug","confidence":0.0,"reason":"short reason","evidence":["short fact"]},
 "data":{"label":"bug|non-bug","confidence":0.0,"reason":"short reason","evidence":["short fact"]},
 "pattern":{"label":"bug|non-bug","confidence":0.0,"reason":"short reason","evidence":["short fact"]}}
Use a numeric confidence from 0.0 to 1.0. Keep each reason under 12 words and evidence to at most one short phrase per role."""


class CouncilAgent(BaseAuditAgent):
    """Produces three role-specific perspectives in one call; they are not independent agents."""

    def __init__(self, model: str = "qwen/qwen3.8-27b", **kwargs):
        kwargs.setdefault("max_completion_tokens", 512)
        super().__init__(name="multi_perspective_council", model=model, **kwargs)

    async def evaluate(self, issue: dict[str, Any]) -> list[AgentVerdict]:
        from src.knowledge_graph.retriever import get_kg_retriever

        kg_evidence = get_kg_retriever().retrieve_evidence_for_issue(issue)
        user_prompt = f"""Project: {issue.get('project', '')} | Source: {issue.get('source', '')}
Title: {issue.get('title', '')}
Text: {str(issue.get('model_text') or issue.get('description') or '')[:1800]}
Metadata: component={issue.get('component')}; severity={issue.get('severity')}; priority={issue.get('priority')}; status={issue.get('status')}; resolution={issue.get('resolution')}
Knowledge graph evidence: {"; ".join(kg_evidence[:4]) or "none available"}
Original source label (context only): {issue.get('original_label', '')}
Return the three perspective objects in the required JSON format."""

        result = await self._generate_json_response(COUNCIL_PROMPT, user_prompt)
        verdicts: list[AgentVerdict] = []
        for perspective, agent_name in (("policy", "policy_agent"), ("data", "data_agent"), ("pattern", "pattern_agent")):
            raw = result.get(perspective)
            if not isinstance(raw, dict):
                raise ValueError(f"Council response is missing the '{perspective}' object")
            label = str(raw.get("label", "")).lower().strip()
            if label not in {"bug", "non-bug"}:
                raise ValueError(f"Council returned invalid {perspective} label: {label!r}")
            verdicts.append(AgentVerdict(
                agent=agent_name,
                label=label,
                confidence=float(raw.get("confidence")),
                reason=str(raw.get("reason", ""))[:500],
                evidence=[str(item)[:300] for item in raw.get("evidence", [])][:2],
            ))
        return verdicts
