# 本地 Qwen3 分词资源

- 来源：https://huggingface.co/Qwen/Qwen3-0.6B
- 固定版本：`c1899de289a04d12100db370d81485cdf75e47ca`
- 文件：上游 `tokenizer.json`，本地命名为 `qwen3.json`，未修改。
- SHA-256：`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
- 许可证：Apache-2.0，同目录 `LICENSE` 为上游许可证。

此文件随源码提供，运行时不联网下载。使用 `tokenizers` 的 `encode` 统计 token，禁用截断和填充，不插入特殊 token。

默认词表不能证明与云端 `qwen3.8-max` 等模型一致，也不模拟供应商的聊天模板、隐藏推理、工具协议开销。因此本地预算和用量是估算，不等同于账单。可用 `GEOAGENT_TOKENIZER_FILE` 或模型配置的 `tokenizer_file` 指向匹配模型的本地文件；词表不存在或无效时应修正配置，不自动换词表。
