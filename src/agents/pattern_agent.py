from __future__ import annotations

from typing import Any
from src.agents.base_agent import BaseAuditAgent
from src.agents.schemas import AgentVerdict

PATTERN_SYSTEM_PROMPT = """You are the Defect Pattern Matching Agent in a Multi-Agent Software Defect Prediction system.
Your mission is to identify linguistic, structural, and technical defect patterns in software issue reports.

Defect Pattern Indicators:
- Exception & Crash Patterns: Stack traces, NullPointer, Segmentation fault, OutOfMemory, Panic, core dump.
- Failure Mode Patterns: "fails to", "unexpected", "throws error", "regression after commit", "breaks when", "infinite loop".
- Reproducibility Patterns: "Steps to reproduce", "Expected vs Actual behavior", "Minimal repro".
- Non-Defect Patterns: "Proposal:", "Feature request:", "Please add support for", "How do I", "Upgrade dependency to", "Clean up unused code".

Instructions:
1. Detect characteristic defect or non-defect patterns in the text.
2. Evaluate pattern strength and prevalence.
3. Output STRICT JSON adhering to this schema:
{
  "agent": "pattern_agent",
  "label": "bug" or "non-bug",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<explanation of identified technical patterns>",
  "evidence": ["<pattern match 1>", "<pattern match 2>"]
}
"""

class PatternAgent(BaseAuditAgent):
    """Uses linguistic patterns and technical signatures to classify issues (AGENTS.md Sec 18.3)."""

    def __init__(self, model: str = "qwen/qwen3.8-27b", **kwargs):
        super().__init__(name="pattern_agent", model=model, **kwargs)

    async def evaluate(self, issue: dict[str, Any]) -> AgentVerdict:
        text = issue.get("model_text", f"{issue.get('title', '')} {issue.get('description', '')}")

        user_prompt = f"""Issue Text:
{text[:2000]}

Scan for defect signatures, crash patterns, or enhancement proposals.
Respond in valid JSON."""

        raw_json = await self._generate_json_response(PATTERN_SYSTEM_PROMPT, user_prompt)
        
        label = str(raw_json.get("label", "bug")).lower().strip()
        if "non" in label:
            norm_label = "non-bug"
        else:
            norm_label = "bug"

        return AgentVerdict(
            agent="pattern_agent",
            label=norm_label,
            confidence=float(raw_json.get("confidence", 0.85)),
            reason=str(raw_json.get("reason", "Evaluated linguistic defect patterns")),
            evidence=list(raw_json.get("evidence", [])),
        )
