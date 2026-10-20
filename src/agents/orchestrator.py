from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import pyarrow.parquet as pq
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

        retries = int(llm_cfg.get("max_retries", 6))
        self.policy_agent = PolicyAgent(model=llm_cfg.get("subagent_model", "qwen/qwen3.8-27b"), max_retries=retries)
        self.data_agent = DataAgent(model=llm_cfg.get("subagent_model", "qwen/qwen3.8-27b"), max_retries=retries)
        self.pattern_agent = PatternAgent(model=llm_cfg.get("subagent_model", "qwen/qwen3.8-27b"), max_retries=retries)
        self.judge_agent = JudgeAgent(model=llm_cfg.get("judge_model", "openai/gpt-oss-120b"), max_retries=retries)

    async def audit_single_issue(self, issue: dict[str, Any]) -> AuditedIssueRecord:
        """Runs the 3 specialized agents concurrently, then arbitrates via the Judge."""
        agents = [self.policy_agent, self.data_agent, self.pattern_agent, self.judge_agent]
        usage_before = [(a.prompt_tokens, a.completion_tokens) for a in agents]
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
            policy_evidence=policy_res.evidence,
            data_label=data_res.label,
            data_confidence=data_res.confidence,
            data_reason=data_res.reason,
            data_evidence=data_res.evidence,
            pattern_label=pattern_res.label,
            pattern_confidence=pattern_res.confidence,
            pattern_reason=pattern_res.reason,
            pattern_evidence=pattern_res.evidence,
            judge_label=judge_res.final_label,
            judge_confidence=judge_res.confidence,
            judge_reasoning=judge_res.reasoning_summary,
            judge_evidence=judge_res.evidence,
            agent_agreement=judge_res.agreement,
            label_flipped=label_flipped,
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
            api_prompt_tokens=sum(a.prompt_tokens - before[0] for a, before in zip(agents, usage_before)),
            api_completion_tokens=sum(a.completion_tokens - before[1] for a, before in zip(agents, usage_before)),
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
    parser.add_argument("--sample-size", type=int, default=0, help="Optional cap; default audits every row in the preselected training-only sample")
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

    # Filter out already audited issues
    parquet_out = Path(orchestrator.config["audit"].get("output_parquet", "data/processed/audited_labels.parquet"))
    if parquet_out.exists():
        try:
            audited_existing = pd.read_parquet(parquet_out, columns=["source", "project", "issue_id"])
            existing_keys = set(zip(audited_existing["source"], audited_existing["project"], audited_existing["issue_id"].astype(str)))
            df = df[~df.apply(lambda row: (row["source"], row["project"], str(row["issue_id"])) in existing_keys, axis=1)]
            print(f"Skipping {len(existing_keys)} already-audited records. Remaining candidate pool: {len(df)}")
        except Exception as e:
            print(f"[Info] Note on existing audited parquet: {e}")

    # The input file is already the reproducible, training-only audit sample.
    if args.sample_size > 0 and args.sample_size < len(df):
        df_sample = df.sample(n=args.sample_size, random_state=args.random_seed)
    else:
        df_sample = df


    issues = df_sample.to_dict(orient="records")
    results = await orchestrator.run_batch_audit(issues, batch_size=args.batch_size)

    agents = [orchestrator.policy_agent, orchestrator.data_agent, orchestrator.pattern_agent, orchestrator.judge_agent]
    prompt_tokens = sum(agent.prompt_tokens for agent in agents)
    completion_tokens = sum(agent.completion_tokens for agent in agents)

    save_audited_records(
        results,
        output_dir=orchestrator.config["audit"].get("output_dir", "outputs/results/audited"),
        parquet_path=orchestrator.config["audit"].get("output_parquet", "data/processed/audited_labels.parquet"),
    )
    usage_path = Path(orchestrator.config["audit"].get("output_dir", "outputs/results/audited")) / "api_usage.json"
    pricing = orchestrator.config["llm"].get("pricing_usd_per_million_tokens", {})
    usage_by_model: dict[str, dict[str, int]] = {}
    for agent in agents:
        for model, usage in agent.usage_by_model.items():
            aggregate = usage_by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0})
            aggregate["prompt_tokens"] += usage["prompt_tokens"]
            aggregate["completion_tokens"] += usage["completion_tokens"]
    estimated_cost = None
    if pricing and all(model in pricing for model in usage_by_model):
        estimated_cost = sum(
            (usage["prompt_tokens"] * float(pricing[model]["input"]) +
             usage["completion_tokens"] * float(pricing[model]["output"])) / 1_000_000
            for model, usage in usage_by_model.items()
        )
    usage_path.write_text(json.dumps({
        "audited_records_this_run": len(results),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "usage_by_model": usage_by_model,
        "estimated_cost_usd": estimated_cost,
        "pricing_source": "configs/agents_config.yaml per-model rates" if estimated_cost is not None else "not configured for every model; token usage only",
    }, indent=2), encoding="utf-8")
    all_audited = pd.read_parquet(parquet_out)
    train_split_path = Path("data/processed/fixed_splits/train.parquet")
    training_records = pq.ParquetFile(train_split_path).metadata.num_rows
    flips = int(all_audited["label_flipped"].sum())
    audit_metrics = {
        "audited_records": int(len(all_audited)),
        "training_records": int(training_records),
        "audit_coverage": float(len(all_audited) / training_records),
        "label_flips": flips,
        "flip_rate": float(flips / len(all_audited)) if len(all_audited) else 0.0,
        "mean_agent_agreement": float(all_audited["agent_agreement"].mean()) if len(all_audited) else None,
        "mean_judge_confidence": float(all_audited["judge_confidence"].mean()) if len(all_audited) else None,
        "api_usage_this_run": json.loads(usage_path.read_text(encoding="utf-8")),
    }
    metrics_path = usage_path.parent / "audit_metrics.json"
    metrics_path.write_text(json.dumps(audit_metrics, indent=2), encoding="utf-8")
    print(f"API token/cost summary written to {usage_path}")
    print("Multi-agent auditing process completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

