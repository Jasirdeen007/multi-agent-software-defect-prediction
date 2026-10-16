import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from backend.main import app



async def main():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r1 = await client.get("/api/v1/health")
        print("Health check:", r1.status_code, r1.json())

        r2 = await client.get("/api/v1/project-status")
        print("\nProject status:", r2.status_code, r2.json())

        r3 = await client.get("/api/v1/audit-overview")
        print("\nAudit overview:", r3.status_code, r3.json())

        r4 = await client.get("/api/v1/model-comparison")
        print("\nModel comparison treatmentStatus:", r4.json().get("treatmentStatus"))
        print("Metrics sample:", r4.json().get("metrics")[:3])

        r5 = await client.get("/api/v1/audit-cases?pageSize=2")
        print("\nAudit cases:", r5.status_code, "Total:", r5.json().get("total"))
        if r5.json().get("cases"):
            first_case_id = r5.json()["cases"][0]["id"]
            print("First case ID:", first_case_id)
            r5_detail = await client.get(f"/api/v1/audit-cases/{first_case_id}")
            print("\nCase detail Knowledge Graph evidence:", r5_detail.json().get("knowledgeGraphEvidence"))

        payload = {
            "title": "NullPointerException in Kafka consumer heartbeat",
            "description": "The consumer thread crashes with SIGSEGV when rebalancing partition assignments.",
        }
        r6 = await client.post("/api/v1/predictions", json=payload)
        print("\nPrediction endpoint:", r6.status_code, r6.json())


if __name__ == "__main__":
    asyncio.run(main())
