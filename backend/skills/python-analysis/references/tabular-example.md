# 表格统计与交付示例

用于已确定分组字段、数值字段和缺失值口径的 CSV 统计。这里演示接口衔接，不是每次 Python 调用的固定模板。

示例将空白分组键统一为缺失组并保留；数值空值不参与求和，全部缺失的组保持空值；非空但不能转换成数值的内容会报错，等待判断。只有实际任务采用这些口径时才复用。

先根据已有证据替换 `dataset_id`、`group_column`、`value_column`，并在工具调用的 `dataset_ids` 中传入同一个真实 ID。编码、分隔符、字段含义不同或数据不能完整装入内存时，需要调整读取与计算方式。

```python
import json
import math

import pandas as pd

dataset_id = "REPLACE_WITH_VERIFIED_DATASET_ID"
group_column = "region_code"
value_column = "amount"

# 从原文件读取标识为字符串，避免丢失前导零。
frame = pd.read_csv(
    datasets[dataset_id]["path"],
    usecols=[group_column, value_column],
    dtype="string",
    keep_default_na=False,
    encoding="utf-8-sig",
)
group_values = frame[group_column].str.strip().replace("", pd.NA)
raw_values = frame[value_column].str.strip().replace("", pd.NA)
numeric_values = pd.to_numeric(raw_values, errors="coerce")
invalid_values = raw_values.notna() & numeric_values.isna()
if invalid_values.any():
    raise ValueError(f"存在 {int(invalid_values.sum())} 条非空数值转换失败记录，需要确定处理口径。")
if not numeric_values.dropna().map(math.isfinite).all():
    raise ValueError("数值字段含无穷大等非有限数值，需要检查输入。")

working = pd.DataFrame({"group_key": group_values, "numeric_value": numeric_values})
grouped = working.groupby("group_key", dropna=False, sort=True)["numeric_value"]
summary = pd.concat(
    [
        grouped.size().rename("row_count"),
        grouped.count().rename("value_count"),
        grouped.sum(min_count=1).rename("value_sum"),
    ],
    axis=1,
).reset_index()

if not summary["value_sum"].dropna().map(math.isfinite).all():
    raise ValueError("分组汇总产生非有限数值，需要调整数值类型或计算方式。")
if int(summary["row_count"].sum()) != len(frame):
    raise ValueError("分组后的记录总数与输入不一致。")
if int(summary["value_count"].sum()) != int(numeric_values.notna().sum()):
    raise ValueError("分组后的有效数值数量与输入不一致。")

output_path = output_dir / "group_summary.csv"
summary.to_csv(output_path, index=False, encoding="utf-8-sig")
saved = pd.read_csv(output_path, dtype={"group_key": "string"}, keep_default_na=False, encoding="utf-8-sig")
if len(saved) != len(summary) or list(saved.columns) != list(summary.columns):
    raise ValueError("导出文件的行数或字段与计算结果不一致。")

print(json.dumps({
    "input_rows": len(frame),
    "missing_group_rows": int(group_values.isna().sum()),
    "missing_value_rows": int(numeric_values.isna().sum()),
    "output_groups": len(summary),
    "output_file": output_path.name,
}, ensure_ascii=False, allow_nan=False))
```

示例处理的是普通数值汇总。精确金额、大整数或指定误差的计算，应选择满足精度要求的数据类型和核对方式。完成调用后使用工具实际返回的资源引用交付文件；上述打印内容不生成资源 ID。
