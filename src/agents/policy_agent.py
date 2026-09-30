from __future__ import annotations

from typing import Any
from src.agents.base_agent import BaseAuditAgent
from src.agents.schemas import AgentVerdict

POLICY_SYSTEM_PROMPT = """You are the Policy Compliance Agent in a Multi-Agent Software Defect Prediction system.
Your mission is to evaluate software issue reports against standardized defect taxonomies (ISO/IEC/IEEE 24765 and BugHub standards).

Standard Definitions:
- BUG: A fault in software that causes an incorrect or unexpected result, runtime crash, memory leak, data corruption, broken functional contract, or failure to perform an expected behavior as specified.
- NON-BUG: A request for new functionality, feature enhancement, architectural refactoring, documentation update, build/dependency upgrade, or developer inquiry without an existing operational failure.

Instructions:
1. Examine the issue title and description objectively.
2. Focus strictly on whether the text describes an observed malfunction vs an enhancement.
3. Output STRICT JSON adhering to this schema:
{
  "agent": "policy_agent",
  "label": "bug" or "non-bug",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<clear explanation citing specific observed behavior vs enhancement>",
  "evidence": ["<specific quote or fact 1>", "<specific quote or fact 2>"]
}
"""

class PolicyAgent(BaseAuditAgent):
    """Evaluates issue reports against explicit software engineering defect policies (AGENTS.md Sec 18.1)."""

    def __init__(self, model: str = "qwen/qwen3.8-27b", **kwargs):
        super().__init__(name="policy_agent", model=model, **kwargs)

    async def evaluate(self, issue: dict[str, Any]) -> AgentVerdict:
        title = issue.get("title", "")
        description = issue.get("description", issue.get("model_text", ""))
        project = issue.get("project", "unknown")

        user_prompt = f"""Software Project: {project}
Issue Title: {title}
Issue Description:
{description}

Evaluate whether this issue describes a genuine software BUG or a NON-BUG according to engineering policy.
Respond in valid JSON."""

        raw_json = await self._generate_json_response(POLICY_SYSTEM_PROMPT, user_prompt)
        
        # Ensure label is normalized
        label = str(raw_json.get("label", "bug")).lower().strip()
        if "non" in label:
            norm_label = "non-bug"
        else:
            norm_label = "bug"

        return AgentVerdict(
            agent="policy_agent",
            label=norm_label,
            confidence=float(raw_json.get("confidence", 0.85)),
            reason=str(raw_json.get("reason", "Evaluated against defect policy")),
            evidence=list(raw_json.get("evidence", [])),
        )
