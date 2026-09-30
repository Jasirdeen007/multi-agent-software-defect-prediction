from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class AgentVerdict(BaseModel):
    """Structured output contract for individual specialized audit agents (AGENTS.md Sec 18)."""
    agent: str = Field(description="Identifier of the agent (policy_agent, data_agent, pattern_agent)")
    label: Literal["bug", "non-bug"] = Field(description="Audited label assigned by the agent")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    reason: str = Field(description="Concise justification for the label decision")
    evidence: list[str] = Field(default_factory=list, description="Specific textual or metadata facts supporting decision")


class JudgeVerdict(BaseModel):
    """Structured output contract for the Judge Agent synthesising sub-agent verdicts (AGENTS.md Sec 19)."""
    final_label: Literal["bug", "non-bug"] = Field(description="Final synthesized label after arbitration")
    confidence: float = Field(ge=0.0, le=1.0, description="Judge confidence in the final label")
    agreement: float = Field(ge=0.0, le=1.0, description="Agreement proportion among sub-agents (1.0 = unanimous, 0.67 = majority)")
    agent_outputs: dict[str, Any] = Field(default_factory=dict, description="Raw outputs from all sub-agents preserved")
    reasoning_summary: str = Field(description="Summary of synthesis and resolution of any conflicts")
    evidence: list[str] = Field(default_factory=list, description="Consolidated key evidence supporting final label")


class AuditedIssueRecord(BaseModel):
    """Complete audit trail record for an issue, strictly preserving original_label (AGENTS.md Sec 14 & 21)."""
    source: str
    project: str
    issue_id: str
    original_label: str
    
    # Sub-agent outputs
    policy_label: str
    policy_confidence: float
    policy_reason: str
    
    data_label: str
    data_confidence: float
    data_reason: str
    
    pattern_label: str
    pattern_confidence: float
    pattern_reason: str
    
    # Judge output
    judge_label: str
    judge_confidence: float
    judge_reasoning: str
    agent_agreement: float
    
    # Analysis flags
    label_flipped: bool
    audit_timestamp: str
