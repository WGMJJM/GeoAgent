# 大模型审核口径

`model_review_results.jsonl` 和 `model_review_summary.json` 由 `scripts/score_model_review.py` 从 `evaluator_original_gis_results.jsonl` 生成。判分只读取 Run 终态和运行时大模型完成审核结论，不调用 `validate_gis.py` 的数值、文件名或地图程序校验。

这项统计是“大模型审核通过的完成率”，不是独立验证的 GIS 计算正确率。等待用户、部分完成、工具执行失败和系统阻塞分别保留，不因移除程序判分而自动算成功。

原始运行记录和旧程序判分结果均保留，新的统计不会覆盖它们。
