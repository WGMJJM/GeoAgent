"""显式导入 ArcGIS .rc / HTML；保留原语言、参数名、版本及来源，不用 LLM 改写。"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

from .store import digest


class DocumentHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored = 0
        self.main_start: int | None = None
        self.main_end: int | None = None

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "head", "footer"}:
            self.ignored += 1
        if tag == "main":
            self.main_start = len(self.parts)
        if not self.ignored:
            if tag in {"p", "div", "li", "br", "tr", "pre"} or re.fullmatch(r"h[1-6]", tag):
                self.parts.append("\n\n")
            # .rc 的 span title 保存实际枚举值，正文常常只有翻译后的标签。
            title = dict(attrs).get("title")
            if title:
                self.parts.append(f" [{title}] ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "head", "footer"}:
            self.ignored -= 1
        if tag == "main":
            self.main_end = len(self.parts)
        if tag in {"p", "div", "li", "tr", "pre"} or re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)

    def text(self) -> str:
        parts = self.parts[self.main_start:self.main_end] if self.main_start is not None else self.parts
        return "\n\n".join(" ".join(part.split()) for part in "".join(parts).split("\n\n") if part.strip())


def plain_text(html: str) -> str:
    parser = DocumentHTML()
    parser.feed(html)
    return parser.text()


def split_text(text: str, header: str, count_tokens, limit: int) -> list[str]:
    """按段/句打包，极长单句才按字符边界切块；每块完整保留文字，没有截断丢弃。"""
    if count_tokens(header + "\n") >= limit:
        raise ValueError("文档标题/参数路径已超过分块预算。")
    result, pending = [], ""
    for paragraph in re.split(r"\n\s*\n", text):
        if not paragraph.strip():
            continue
        if count_tokens(header + "\n" + paragraph) <= limit:
            units = [paragraph]
        else:
            units = re.split(r"(?<=[。！？])|(?<=[.!?])\s+", paragraph)
        for unit in units:
            candidate = pending + ("\n\n" if pending else "") + unit
            if count_tokens(header + "\n" + candidate) <= limit:
                pending = candidate
                continue
            if pending:
                result.append(header + "\n" + pending)
                pending = ""
            while unit and count_tokens(header + "\n" + unit) > limit:
                lo, hi = 1, len(unit)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if count_tokens(header + "\n" + unit[:mid]) <= limit:
                        lo = mid
                    else:
                        hi = mid - 1
                if count_tokens(header + "\n" + unit[:lo]) > limit:
                    raise ValueError("分块预算不足以保留标题和正文字符，请提高分块限额。")
                result.append(header + "\n" + unit[:lo])
                unit = unit[lo:]
            pending = unit
    if pending:
        result.append(header + "\n" + pending)
    return result


def read_rc(path: Path, *, definitions_root: Path | None = None) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))["map"]
    # 部分语言包用空 map 字符串表示尚未提供翻译，不是一篇空白技术文档。
    if data == "" or data == {}:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"不支持的资源 map 格式：{path}")
    toolbox, tool = path.parent.parent, path.parent.stem
    definitions = definitions_root / toolbox.name if definitions_root else toolbox
    descriptor = definitions / "toolbox.content"
    tool_name = ""
    if descriptor.is_file():
        content = json.loads(descriptor.read_text(encoding="utf-8-sig"))
        for group in content.get("toolsets", {}).values():
            for entry in group.get("tools", []):
                name, _, folder = entry.partition(":")
                if folder == path.parent.name:
                    tool = name
        if content.get("alias"):
            tool_name = f"arcpy.{tool}_{content['alias']}".lower()
    keywords = path.with_name("tool.keywords.rc")
    keyword_data = json.loads(keywords.read_text(encoding="utf-8-sig")).get("set", []) if keywords.is_file() else []
    sections = [("overview", plain_text(data.get("summary") or data.get("description", "")))]
    for key, value in data.items():
        if key.endswith(".descr"):
            parameter = key.removesuffix(".descr")
            sections.append((parameter, f"{data.get(parameter + '.title', parameter)}\n\n{plain_text(value)}"))
    return {"title": data.get("title", tool), "tool_name": tool_name,
            "family": f"{toolbox.name}/{tool}", "keywords": keyword_data,
            "sections": [(section, text) for section, text in sections if text.strip()]}


def import_documents(store, *, source: Path, collection: str, product: str, version: str,
                     language: str, kind: str, count_tokens, chunk_tokens: int,
                     definitions_root: Path | None = None, source_url: str | None = None) -> dict:
    source = source.resolve()
    if not source.exists():
        raise ValueError(f"资料路径不存在：{source}")
    files = [source] if source.is_file() else sorted(
        source.rglob("tool.content.rc") if kind == "rc" else
        (path for path in source.rglob("*") if path.suffix.lower() in {".html", ".htm"}))
    if not files:
        raise ValueError("没有找到可导入的资料；原有知识库未修改。")
    chunks, empty_documents = [], 0
    for path in files:
        doc = read_rc(path, definitions_root=definitions_root) if kind == "rc" else {
            "title": path.stem, "tool_name": "", "family": str(path.relative_to(source.parent if source.is_file() else source)),
            "keywords": [], "sections": [("document", plain_text(path.read_text(encoding="utf-8-sig")))],
        }
        if doc is None or not doc["sections"] or not any(text.strip() for _, text in doc["sections"]):
            empty_documents += 1
            continue
        source_hash = digest(path.read_text(encoding="utf-8-sig"))
        reference = str(path)
        if source_url:
            reference = source_url if source.is_file() else source_url.rstrip("/") + "/" + quote(path.relative_to(source).as_posix())
        document_id = digest(collection + "|" + str(path))
        header = f"{product} {version} | {doc['tool_name'] or doc['family']} | {doc['title']}"
        whole = "\n\n".join(f"{section}\n{text}" for section, text in doc["sections"])
        sections = [("document", whole)] if count_tokens(header + "\n" + whole) <= chunk_tokens else doc["sections"]
        for section, body in sections:
            for ordinal, text in enumerate(split_text(body, header + " | " + section, count_tokens, chunk_tokens)):
                family = f"{product}|{version}|{doc['family']}|{section}|{ordinal}"
                chunks.append({"id": "chunk_" + digest(f"{document_id}|{section}|{ordinal}")[:24],
                               "document_id": document_id, "title": doc["title"], "tool_name": doc["tool_name"],
                               "section": section, "family": family, "product": product, "version": version,
                               "language": language, "source": reference, "source_hash": source_hash,
                               "keywords": doc["keywords"], "text": text, "content_hash": digest(text)})
    if not chunks:
        raise ValueError("资料中没有可用正文；原有知识库未修改。")
    result = store.replace_collection(collection, chunks)
    return {"documents": len(files) - empty_documents, "empty_documents": empty_documents, **result}
