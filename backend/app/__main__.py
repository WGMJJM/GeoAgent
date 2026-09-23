"""GeoAgent CLI。"""

from __future__ import annotations

import argparse
import asyncio
import json

from app.api.app import create_app
from app.application import Application
from app.demo import seed_demo
from evaluation import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(prog="geoagent", description="GIS-oriented Agent Harness")
    parser.add_argument("--init", action="store_true", help="初始化 SQLite 和 workspace")
    parser.add_argument("--demo", action="store_true", help="生成演示数据并运行两条验收场景")
    parser.add_argument("--query", help="执行一次自然语言 GIS 请求")
    parser.add_argument("--serve", action="store_true", help="启动 FastAPI 服务")
    parser.add_argument("--evaluate", action="store_true", help="运行离线 GIS 评测用例")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    application = Application()
    application.start()
    if args.init:
        print(f"GeoAgent state: {application.store.database_path}")
        print(f"GeoAgent workspace: {application.workspace.root}")
    if args.demo:
        ids = seed_demo(application)
        print(json.dumps({"datasets": ids}, ensure_ascii=False, indent=2))
        asyncio.run(_run_demo(application, ids))
    if args.query:
        result = asyncio.run(application.ask(args.query))
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if args.evaluate:
        summary = asyncio.run(EvaluationRunner(application).run())
        payload = summary.model_dump(mode="json")
        payload["失败用例"] = [
            {"用例": item.case_id, "失败原因": item.failures}
            for item in summary.cases
            if not item.passed
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.serve:
        import uvicorn

        uvicorn.run(create_app(application), host=args.host, port=args.port)


async def _run_demo(application: Application, ids: dict[str, str]) -> None:
    first = await application.ask("检查 roads 并生成 500 米缓冲区", dataset_ids=[ids["roads"]])
    print(json.dumps({"buffer_case": first.model_dump(mode="json")}, ensure_ascii=False, indent=2))
    second = await application.ask("综合道路、人口和 DEM，从三个方面评价当前区域", dataset_ids=list(ids.values()))
    print(json.dumps({"multi_agent_case": second.model_dump(mode="json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
