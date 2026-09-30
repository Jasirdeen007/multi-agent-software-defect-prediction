from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import yaml

from src.agents.data_agent import DataAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.pattern_agent import PatternAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.schemas import AuditedIssueRecord
from src.agents.storage import save_audited_records


class MultiAgentAuditOrchestrator:
    """Orchestrates parallel multi-agent auditing of software issue labels (AGENTS.md Phase II)."""

    def __init__(self, config_path: str = "configs/agents_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        llm_cfg = self.config["llm"]
        self.rate_limit_delay = float(llm_cfg.get("rate_limit_delay_seconds", 1.5))

        self.policy_agent = PolicyAgent(model=llm_cfg.get("subagent_model", "qwen/qwen3.8-27b"))
        self.data_agent = DataAgent(model=llm_cfg.get("subagent_model", "qwen/qwen3.8-27b"))
        self.pattern_agent = PatternAgent(model=llm_cfg.get("subagent_model", "qwen/qwen3.8-27b"))
        self.judge_agent = JudgeAgent(model=llm_cfg.get("judge_model", "openai/gpt-oss-120b"))

    async def audit_single_issue(self, issue: dict[str, Any]) -> AuditedIssueRecord:
        """Runs the 3 specialized agents concurrently, then arbitrates via the Judge."""
        # 1. Run Policy, Data, and Pattern agents in parallel
        verdicts = await asyncio.gather(
            self.policy_agent.evaluate(issue),
            self.data_agent.evaluate(issue),
            self.pattern_agent.evaluate(issue),
        )
        policy_res, data_res, pattern_res = verdicts

        # 2. Arbitrate with Judge
        judge_res = await self.judge_agent.arbitrate(issue, list(verdicts))

        # 3. Assess label flip against original BugHub source label
        orig_label = str(issue.get("original_label", "")).lower().strip()
        norm_orig = "non-bug" if "non" in orig_label else "bug"
        label_flipped = (judge_res.final_label != norm_orig)

        return AuditedIssueRecord(
            source=str(issue.get("source", "")),
            project=str(issue.get("project", "")),
            issue_id=str(issue.get("issue_id", "")),
            original_label=norm_orig,
            policy_label=policy_res.label,
            policy_confidence=policy_res.confidence,
            policy_reason=policy_res.reason,
            data_label=data_res.label,
            data_confidence=data_res.confidence,
            data_reason=data_res.reason,
            pattern_label=pattern_res.label,
            pattern_confidence=pattern_res.confidence,
            pattern_reason=pattern_res.reason,
            judge_label=judge_res.final_label,
            judge_confidence=judge_res.confidence,
            judge_reasoning=judge_res.reasoning_summary,
            agent_agreement=judge_res.agreement,
            label_flipped=label_flipped,
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def run_batch_audit(
        self,
        issues: list[dict[str, Any]],
        batch_size: int = 2,
        checkpoint_freq: int = 25,
    ) -> list[AuditedIssueRecord]:
        """Processes a list of issues in controlled batches with rate limit pacing and checkpointing."""
        audited_records: list[AuditedIssueRecord] = []
        total = len(issues)
        print(f"Starting multi-agent audit on {total} issues (batch size = {batch_size})...")

        semaphore = asyncio.Semaphore(2)  # max 2 simultaneous issue evaluations

        async def _bounded_audit(issue: dict[str, Any]):
            async with semaphore:
                return await self.audit_single_issue(issue)

        for i in range(0, total, batch_size):
            batch = issues[i : i + batch_size]
            tasks = [_bounded_audit(issue) for issue in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in batch_results:
                if isinstance(res, Exception):
                    print(f"[Warning] Failed to audit issue: {res}")
                else:
                    audited_records.append(res)

            print(f"Audited {len(audited_records)}/{total} issues...")

            # Periodic checkpoint
            if len(audited_records) % checkpoint_freq == 0 and len(audited_records) > 0:
                save_audited_records(
                    audited_records,
                    output_dir=self.config["audit"].get("output_dir", "outputs/results/audited"),
                    parquet_path=self.config["audit"].get("output_parquet", "data/processed/audited_labels.parquet"),
                )

            # Rate limit pacing
            if i + batch_size < total:
                await asyncio.sleep(self.rate_limit_delay)

        return audited_records


async def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Defect Label Auditing Runner")
    parser.add_argument("--config", type=str, default="configs/agents_config.yaml")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of issues to audit")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    orchestrator = MultiAgentAuditOrchestrator(config_path=args.config)

    dataset_path = orchestrator.config["audit"]["dataset_path"]
    print(f"Loading {dataset_path} for audit...")
    cols = [
        "source", "project", "issue_id", "title", "description",
        "model_text", "component", "severity", "priority", "status", "resolution", "original_label"
    ]
    df = pd.read_parquet(dataset_path, columns=cols)

    # Sample issues
    if args.sample_size and args.sample_size < len(df):
        df_sample = df.sample(n=args.sample_size, random_state=args.random_seed)
    else:
        df_sample = df

    issues = df_sample.to_dict(orient="records")
    results = await orchestrator.run_batch_audit(issues, batch_size=args.batch_size)

    save_audited_records(
        results,
        output_dir=orchestrator.config["audit"].get("output_dir", "outputs/results/audited"),
        parquet_path=orchestrator.config["audit"].get("output_parquet", "data/processed/audited_labels.parquet"),
    )
    print("Multi-agent auditing process completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

