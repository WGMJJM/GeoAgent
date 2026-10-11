"""本机管理员入口：import-rc / import-html / embed / search / status。"""

import argparse
import asyncio
import json
from pathlib import Path

from app.config import Settings

from .ingest import import_documents
from .retrieval import KnowledgeService


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[2] / ".env")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("import-rc", "import-html"):
        importer = sub.add_parser(command)
        importer.add_argument("source", type=Path)
        for name in ("collection", "product", "version", "language"):
            importer.add_argument("--" + name, required=True)
        importer.add_argument("--definitions-root", type=Path)
        importer.add_argument("--source-url", help="可选：单文件原始 URL 或目录对应的原始网站根 URL；不发起下载。")
    sub.add_parser("embed")
    sub.add_parser("status")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--tool-name")
    search.add_argument("--product")
    search.add_argument("--version")
    args = parser.parse_args()
    settings = Settings(_env_file=args.env_file)
    service = KnowledgeService(settings)
    try:
        if args.command.startswith("import-"):
            result = import_documents(
                service.store, source=args.source, collection=args.collection, product=args.product,
                version=args.version, language=args.language, kind=args.command.removeprefix("import-"),
                count_tokens=service.embeddings.count_tokens,
                chunk_tokens=min(settings.knowledge_chunk_tokens, settings.embedding_max_input_tokens),
                definitions_root=args.definitions_root,
                source_url=args.source_url,
            )
        elif args.command == "embed":
            result = await service.build_embeddings()
        elif args.command == "search":
            result = await service.search(args.query, tool_name=args.tool_name, product=args.product, version=args.version)
        else:
            revision, chunks = service.store.snapshot()
            key = service.embeddings.key if service.embeddings.configured else None
            result = {"revision": revision, "documents": len({item['document_id'] for item in chunks}),
                      "chunks": len(chunks), "embedding_configured": service.embeddings.configured,
                      "matching_vectors": sum(item['embedding_key'] == key and item['vector'] is not None for item in chunks),
                      "path": str(service.store.path)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
