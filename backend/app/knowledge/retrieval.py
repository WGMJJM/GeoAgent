"""BM25 + 向量检索，以 RRF 融合；结果直接携带原文片段而非只有 ID。"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict, defaultdict
from threading import RLock

import bm25s
import httpx
import numpy as np

from app.core.search import tokenize_terms
from app.core.tokens import estimate_tokens

from .embeddings import EmbeddingClient
from .store import KnowledgeStore


class KnowledgeService:
    def __init__(self, settings, metrics=None) -> None:
        self.settings, self.metrics = settings, metrics
        self.store = KnowledgeStore(settings.knowledge_path)
        self.embeddings = EmbeddingClient(settings)
        self._lock = RLock()
        self._snapshot = (-1, [], None)
        self._query_vectors: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    @property
    def available(self) -> bool:
        return self.settings.knowledge_enabled and self.store.revision() > 0

    def _load(self):
        with self._lock:
            if self._snapshot[0] != self.store.revision():
                revision, chunks = self.store.snapshot()
                index = None
                corpus = [list(tokenize_terms(" ".join([item["text"], *item["keywords"]]))) for item in chunks]
                if any(corpus):
                    index = bm25s.BM25(method="lucene")
                    index.index(corpus, show_progress=False)
                self._snapshot = revision, chunks, index
            return self._snapshot

    async def search(self, query: str, *, tool_name: str | None = None,
                     product: str | None = None, version: str | None = None, count_tokens=None) -> dict:
        start = time.perf_counter()
        count_tokens = count_tokens or (lambda text: estimate_tokens(text, self.settings.tokenizer_file))
        _, chunks, index = await asyncio.to_thread(self._load)
        eligible = {i for i, chunk in enumerate(chunks)
                    if (not tool_name or chunk["tool_name"].casefold() == tool_name.casefold())
                    and (not product or chunk["product"].casefold() == product.casefold())
                    and (not version or chunk["version"] == version)}
        payload = {"mode": "bm25", "matches": [], "notice": "资料仅供参考，不证明操作已执行；无匹配不表示能力不存在。"}
        ranks = []
        terms = list(tokenize_terms(query))
        if terms and index is not None:
            scores = await asyncio.to_thread(index.get_scores, terms)
            ranks.append(sorted((i for i in eligible if scores[i] > 0),
                                key=lambda i: (-scores[i], chunks[i]["id"]))[:self.settings.knowledge_bm25_candidates])
        usage = {"requests": 0, "input_tokens": 0, "duration_ms": 0}
        if eligible and self.embeddings.configured:
            key = self.embeddings.key
            vector_ids = [i for i in sorted(eligible) if chunks[i]["embedding_key"] == key and chunks[i]["vector"] is not None]
            if len(vector_ids) == len(eligible):
                try:
                    cache_key = (key, query)
                    query_vector = self._query_vectors.get(cache_key)
                    if query_vector is None:
                        query_vector, usage = await self.embeddings.embed([query])
                        self._query_vectors[cache_key] = query_vector
                        if len(self._query_vectors) > self.settings.knowledge_query_cache_size:
                            self._query_vectors.popitem(last=False)
                    else:
                        self._query_vectors.move_to_end(cache_key)
                        usage["cache_hit"] = True
                    matrix = np.stack([chunks[i]["vector"] for i in vector_ids])
                    if matrix.shape[1] != query_vector.shape[1]:
                        raise ValueError("嵌入维度变化，需要重新构建索引。")
                    scores = (matrix @ query_vector[0]) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector[0]))
                    ranked = sorted(range(len(vector_ids)), key=lambda i: (-scores[i], chunks[vector_ids[i]]["id"]))
                    ranks.append([vector_ids[i] for i in ranked[:self.settings.knowledge_vector_candidates]])
                    payload["mode"] = "hybrid"
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    # 配置错误/网络故障都明确降级；不把异常中的 URL/响应体或凭据发给模型。
                    payload["degraded_reason"] = "EMBEDDING_REQUEST_FAILED"
                    usage = {"requests": None, "input_tokens": None, "duration_ms": None}
            else:
                payload["degraded_reason"] = "EMBEDDING_INDEX_INCOMPLETE"
        elif eligible:
            payload["degraded_reason"] = "EMBEDDING_NOT_CONFIGURED"
        fused = defaultdict(float)
        for branch in ranks:
            for rank, identifier in enumerate(branch, start=1):
                fused[identifier] += 1 / (self.settings.knowledge_rrf_k + rank)
        seen = set()
        payload["embedding_usage"] = usage
        for identifier in sorted(fused, key=lambda i: (-fused[i], chunks[i]["id"])):
            chunk = chunks[identifier]
            if chunk["family"] in seen or chunk["content_hash"] in seen:
                continue
            match = {key: chunk[key] for key in ("id", "title", "tool_name", "section", "product", "version", "language", "source", "text")}
            candidate = {**payload, "matches": [*payload["matches"], match]}
            if count_tokens(json.dumps(candidate, ensure_ascii=False)) > self.settings.knowledge_result_tokens:
                continue
            payload["matches"].append(match)
            seen.update((chunk["family"], chunk["content_hash"]))
            if len(payload["matches"]) >= self.settings.knowledge_results:
                break
        if self.metrics:
            self.metrics.increment("knowledge.searches")
            self.metrics.increment("knowledge.embedding_input_tokens", usage["input_tokens"] or 0)
            self.metrics.increment("knowledge.embedding_requests", usage["requests"] or 0)
            if usage.get("cache_hit"):
                self.metrics.increment("knowledge.query_cache_hits")
            if usage["requests"] is None:
                self.metrics.increment("knowledge.embedding_failed_searches")
            self.metrics.observe("knowledge.search_ms", (time.perf_counter() - start) * 1000)
        return payload

    async def build_embeddings(self) -> dict:
        revision, chunks = self.store.snapshot()
        if not chunks:
            raise ValueError("知识库为空，请先导入资料。")
        if not self.embeddings.configured:
            raise ValueError("请填写独立嵌入配置及 tokenizer 后再构建向量。")
        key = self.embeddings.key
        pending = [item for item in chunks if item["embedding_key"] != key or item["vector"] is None]
        dimensions = {len(item["vector"]) for item in chunks if item["embedding_key"] == key and item["vector"] is not None}
        if len(dimensions) > 1:
            raise ValueError("已有索引维度不一致，请用新的模型配置完整重建。")
        vectors, tokens, requests, dimension = {}, 0, 0, next(iter(dimensions), None)
        for start in range(0, len(pending), self.settings.embedding_batch_size):
            batch = pending[start:start + self.settings.embedding_batch_size]
            encoded, usage = await self.embeddings.embed([item["text"] for item in batch])
            if dimension is not None and encoded.shape[1] != dimension:
                raise ValueError("同一次构建返回了不同维度；旧索引未覆盖。")
            dimension = encoded.shape[1]
            vectors.update({item["id"]: vector for item, vector in zip(batch, encoded, strict=True)})
            tokens += usage["input_tokens"] or 0
            requests += usage["requests"]
        self.store.publish_vectors(revision, key, vectors)
        return {"embedded_chunks": len(vectors), "reused_chunks": len(chunks) - len(vectors),
                "input_tokens": tokens, "requests": requests, "dimension": dimension}

    async def close(self) -> None:
        await self.embeddings.close()
