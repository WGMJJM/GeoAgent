# 本地分词资源

## Qwen3

- 来源：https://huggingface.co/Qwen/Qwen3-0.6B
- 固定版本：`c1899de289a04d12100db370d81485cdf75e47ca`
- 文件：上游 `tokenizer.json`，本地命名为 `qwen3.json`，未修改。
- SHA-256：`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
- 许可证：Apache-2.0，同目录 `LICENSE` 为上游许可证。

## DeepSeek V4.1 Flash

- 来源：https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- 固定版本：`dba1be0a40aa45a94ad051997016db3960a90277`
- 文件：上游 `tokenizer.json`，本地命名为 `deepseek-v4.1.json`，未修改。
- SHA-256：`c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`
- 许可证：MIT，同目录 `deepseek-v4.1-LICENSE` 为上游许可证。

这些文件随源码提供，运行时不联网下载。使用 `tokenizers` 的 `encode` 统计 token，禁用截断和填充，不插入特殊 token。模型 Profile 通过 `tokenizer_file` 选择对应词表，切换模型时会同时切换本地计数器。

本地词表不模拟供应商的聊天模板、隐藏推理和工具协议开销，因此本地预算和用量仍是估算，不等同于账单。可用 `GEOAGENT_TOKENIZER_FILE` 设置全局默认值，并用模型配置的 `tokenizer_file` 覆盖；词表不存在或无效时应修正配置，不自动换词表。
