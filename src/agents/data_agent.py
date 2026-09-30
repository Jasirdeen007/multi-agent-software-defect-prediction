from __future__ import annotations

from typing import Any
from src.agents.base_agent import BaseAuditAgent
from src.agents.schemas import AgentVerdict

DATA_SYSTEM_PROMPT = """You are the Data & Metadata Consistency Agent in a Multi-Agent Software Defect Prediction system.
Your mission is to inspect structured issue metadata and assess whether the metadata signals corroborate a defect report.

Key Metadata Signals:
- Source & Project context: Bugzilla, GitHub, JIRA
- Component: Is it a core operational module or docs/packaging?
- Severity / Priority: Blocker, Critical, Major, High vs Low, Trivial, Wishlist, Enhancement
- Resolution / Status: Resolved (Fixed, Duplicate) vs Won't Fix, Invalid, Worksforme
- Context consistency: Does high severity match the text, or does resolution suggest it was not a genuine defect?

Instructions:
1. Cross-reference the metadata with the issue title.
2. Note that missing metadata (NULLs) is normal for sources like GitHub and must not be treated as corruption.
3. Output STRICT JSON adhering to this schema:
{
  "agent": "data_agent",
  "label": "bug" or "non-bug",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<clear explanation analyzing metadata consistency>",
  "evidence": ["<metadata fact 1>", "<metadata fact 2>"]
}
"""

class DataAgent(BaseAuditAgent):
    """Inspects structured metadata and cross-field consistency (AGENTS.md Sec 18.2)."""

    def __init__(self, model: str = "qwen/qwen3.8-27b", **kwargs):
        super().__init__(name="data_agent", model=model, **kwargs)

    async def evaluate(self, issue: dict[str, Any]) -> AgentVerdict:
        metadata_summary = {
            "source": issue.get("source"),
            "project": issue.get("project"),
            "component": issue.get("component"),
            "severity": issue.get("severity"),
            "priority": issue.get("priority"),
            "status": issue.get("status"),
            "resolution": issue.get("resolution"),
        }
        title = issue.get("title", "")

        user_prompt = f"""Issue Title: {title}
Structured Metadata:
{metadata_summary}

Analyze the metadata indicators. Does the metadata support classifying this issue as a BUG or a NON-BUG?
Respond in valid JSON."""

        raw_json = await self._generate_json_response(DATA_SYSTEM_PROMPT, user_prompt)
        
        label = str(raw_json.get("label", "bug")).lower().strip()
        if "non" in label:
            norm_label = "non-bug"
        else:
            norm_label = "bug"

        return AgentVerdict(
            agent="data_agent",
            label=norm_label,
            confidence=float(raw_json.get("confidence", 0.80)),
            reason=str(raw_json.get("reason", "Evaluated metadata consistency")),
            evidence=list(raw_json.get("evidence", [])),
        )
