"""工具目录和文档检索共用的中英文词项切分，不做语义分类。"""

import re

from bm25s.stopwords import STOPWORDS_EN


def tokenize_terms(text: str) -> tuple[str, ...]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    terms = [item.casefold() for item in re.findall(r"[a-z0-9]+", separated, re.I)
             if len(item) >= 2 and item.casefold() not in STOPWORDS_EN]
    for value in re.findall(r"[\u3400-\u9fff]+", text):
        terms.extend([value] if len(value) <= 2 else
                     (value[index:index + 2] for index in range(len(value) - 1)))
    return tuple(terms)
