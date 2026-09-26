"""本地分词预算计数；请求模板与云端模型不同仍属于估算，不替代 API usage。"""

from functools import cache
from pathlib import Path

from tokenizers import Tokenizer

DEFAULT_TOKENIZER_FILE = Path(__file__).resolve().parents[2] / "resources/tokenizers/qwen3.json"


@cache
def _load_tokenizer(path: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    # 预算计数必须看到整个输入，不能沿用词表文件中的截断或填充设置。
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def estimate_tokens(value: str, tokenizer_file: Path = DEFAULT_TOKENIZER_FILE) -> int:
    return len(_load_tokenizer(tokenizer_file.resolve()).encode(value, add_special_tokens=False).ids)
