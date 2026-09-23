"""不依赖第三方库的结构化日志辅助。"""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

