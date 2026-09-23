"""运行期轻量计数器与阶段耗时观测。"""

from collections import Counter


class Metrics:
    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._timings: dict[str, dict[str, float]] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        self._counts[name] += amount

    def observe(self, name: str, duration_ms: float) -> None:
        """记录阶段最近一次、累计和最大耗时，便于诊断单次运行瓶颈。"""

        duration = max(0.0, float(duration_ms))
        current = self._timings.setdefault(name, {"total_ms": 0.0, "count": 0.0, "max_ms": 0.0, "last_ms": 0.0})
        current["total_ms"] += duration
        current["count"] += 1
        current["max_ms"] = max(current["max_ms"], duration)
        current["last_ms"] = duration

    def snapshot(self) -> dict[str, int | float]:
        snapshot: dict[str, int | float] = dict(self._counts)
        for name, values in self._timings.items():
            # 直接暴露阶段名，满足运行诊断读取最近一次耗时的需要；
            # 聚合字段保留累计、次数和最大值，避免只看到一个无法解释的数字。
            snapshot[name] = round(values["last_ms"], 2)
            snapshot[f"{name}.total_ms"] = round(values["total_ms"], 2)
            snapshot[f"{name}.count"] = int(values["count"])
            snapshot[f"{name}.max_ms"] = round(values["max_ms"], 2)
        return snapshot
