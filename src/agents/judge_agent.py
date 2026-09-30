from __future__ import annotations

from typing import Any
from src.agents.base_agent import BaseAuditAgent
from src.agents.schemas import AgentVerdict, JudgeVerdict

JUDGE_SYSTEM_PROMPT = """You are the Arbitration & Synthesis Judge in a Multi-Agent Software Defect Prediction system.
Your mission is to evaluate individual decisions from the Policy Agent, Data Agent, and Pattern Agent, synthesize their findings, resolve any disagreements, and determine the final audited ground truth label.

Guidelines:
1. Examine all three agent verdicts and their respective reasons and evidence.
2. If all 3 agents agree, confirm the unanimous verdict and summarize the core rationale.
3. If agents disagree (e.g. 2 vs 1):
   - Weigh the strength of textual evidence against metadata signals.
   - Resolve conflicts with clear justification.
4. Output STRICT JSON adhering to this schema:
{
  "final_label": "bug" or "non-bug",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning_summary": "<clear explanation of why this verdict was reached and how disagreement was resolved>",
  "evidence": ["<consolidated decisive evidence 1>", "<consolidated decisive evidence 2>"]
}
"""

class JudgeAgent(BaseAuditAgent):
    """Synthesizes sub-agent verdicts and arbitrates label disputes (AGENTS.md Sec 19 & 21)."""

    def __init__(self, model: str = "openai/gpt-oss-120b", **kwargs):
        super().__init__(name="judge_agent", model=model, **kwargs)

    async def arbitrate(
        self,
        issue: dict[str, Any],
        verdicts: list[AgentVerdict],
    ) -> JudgeVerdict:
        # Calculate quantitative agreement among sub-agents
        labels = [v.label for v in verdicts]
        bug_count = labels.count("bug")
        non_bug_count = labels.count("non-bug")
        majority_count = max(bug_count, non_bug_count)
        agreement_ratio = round(majority_count / len(verdicts), 2)  # 1.0 for 3/3, 0.67 for 2/3

        agent_outputs = {v.agent: v.model_dump() for v in verdicts}

        user_prompt = f"""Target Issue:
Title: {issue.get('title', '')}
Source: {issue.get('source', '')} | Project: {issue.get('project', '')}
Original BugHub Source Label: {issue.get('original_label', 'unknown')}

Agent Verdicts:
1. Policy Agent: {agent_outputs.get('policy_agent', {}).get('label')} (Confidence: {agent_outputs.get('policy_agent', {}).get('confidence')})
   Reason: {agent_outputs.get('policy_agent', {}).get('reason')}
2. Data Agent: {agent_outputs.get('data_agent', {}).get('label')} (Confidence: {agent_outputs.get('data_agent', {}).get('confidence')})
   Reason: {agent_outputs.get('data_agent', {}).get('reason')}
3. Pattern Agent: {agent_outputs.get('pattern_agent', {}).get('label')} (Confidence: {agent_outputs.get('pattern_agent', {}).get('confidence')})
   Reason: {agent_outputs.get('pattern_agent', {}).get('reason')}

Sub-Agent Agreement Ratio: {agreement_ratio} ({majority_count}/{len(verdicts)})

Arbitrate and synthesize these findings. Decide the final audited label.
Respond in valid JSON."""

        raw_json = await self._generate_json_response(JUDGE_SYSTEM_PROMPT, user_prompt)

        final_label = str(raw_json.get("final_label", "bug")).lower().strip()
        if "non" in final_label:
            norm_label = "non-bug"
        else:
            norm_label = "bug"

        return JudgeVerdict(
            final_label=norm_label,
            confidence=float(raw_json.get("confidence", 0.90)),
            agreement=agreement_ratio,
            agent_outputs=agent_outputs,
            reasoning_summary=str(raw_json.get("reasoning_summary", "Synthesized multi-agent consensus")),
            evidence=list(raw_json.get("evidence", [])),
        )
