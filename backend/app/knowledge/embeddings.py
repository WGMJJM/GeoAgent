"""独立的兼容 /embeddings 客户端；不读取聊天模型密钥，不下载模型权重。"""

from __future__ import annotations

import asyncio
import hashlib
import ssl
import time
from functools import cached_property

import httpx
import numpy as np
from tokenizers import Tokenizer

from app.run.recovery import retry_after, transient_error

from .store import digest


class EmbeddingClient:
    def __init__(self, settings, *, transport=None) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=settings.embedding_timeout_seconds,
                                        verify=ssl.create_default_context(), transport=transport)
        self._tokenizer: Tokenizer | None = None

    @property
    def configured(self) -> bool:
        return bool(self.settings.embedding_api_key and self.settings.embedding_model
                    and self.settings.embedding_base_url and self.settings.embedding_tokenizer_path.is_file())

    @cached_property
    def key(self) -> str:
        # 分词器及供应商改变都必须重新生成向量，不能混用向量空间。
        fingerprint = hashlib.sha256(self.settings.embedding_tokenizer_path.read_bytes()).hexdigest()
        return digest(f"{self.settings.embedding_base_url.rstrip('/')}|{self.settings.embedding_model}|{fingerprint}")

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_file(str(self.settings.embedding_tokenizer_path))
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
        return len(self._tokenizer.encode(text, add_special_tokens=True).ids)

    async def embed(self, texts: list[str]) -> tuple[np.ndarray, dict]:
        if not self.configured:
            raise ValueError("请独立配置嵌入模型 URL、模型、密钥及匹配的 tokenizer 文件。")
        if not texts or any(not text.strip() or self.count_tokens(text) > self.settings.embedding_max_input_tokens for text in texts):
            raise ValueError("嵌入输入为空或超过嵌入模型限额；未截断原文，也未发送请求。")
        start = time.perf_counter()
        for attempt in range(self.settings.embedding_max_retries + 1):
            try:
                response = await self._client.post(
                    self.settings.embedding_base_url.rstrip("/") + "/embeddings",
                    headers={"Authorization": "Bearer " + self.settings.embedding_api_key.get_secret_value()},
                    json={"model": self.settings.embedding_model, "input": texts, "encoding_format": "float"},
                )
                response.raise_for_status()
                break
            except httpx.HTTPError as exc:
                if attempt == self.settings.embedding_max_retries or not transient_error(exc):
                    raise
                await asyncio.sleep(min(self.settings.retry_max_delay_seconds,
                                        retry_after(exc) or self.settings.retry_delay_seconds * 2 ** attempt))
        data = response.json()
        items = sorted(data["data"], key=lambda item: item["index"])
        if [item["index"] for item in items] != list(range(len(texts))):
            raise ValueError("嵌入响应条目与请求不一致。")
        vectors = np.asarray([item["embedding"] for item in items], dtype=np.float32)
        if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("嵌入响应包含无效向量。")
        return vectors, {"requests": attempt + 1, "input_tokens": data.get("usage", {}).get("prompt_tokens"),
                         "duration_ms": round((time.perf_counter() - start) * 1000, 2)}

    async def close(self) -> None:
        await self._client.aclose()
