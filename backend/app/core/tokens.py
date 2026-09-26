"""复用会话摘要的轻量 token 估算，不代表模型 tokenizer 的精确计数。"""

import re
from math import ceil


def estimate_tokens(value: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", value))
    other = max(0, len(value) - cjk)
    return max(1, cjk + ceil(other / 4))
