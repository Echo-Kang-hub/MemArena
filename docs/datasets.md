# 数据集使用指南

## 1. 每个测试是否独立？
默认建议独立运行。
- 在批量运行与内置数据集运行中，`isolate_sessions=true` 时，每个 case 会自动生成独立 session_id。
- 若关闭独立会话，则会复用 case 中的 session_id，可能产生记忆串扰。

## 2. 内置数据集放哪里？
请放在目录：`backend/datasets/`
- 文件格式：`*.json`
- 内容格式：JSON 数组，每个元素包含：
  - `case_id`（可选）
  - `session_id`（可选）
  - `input_text`（必填）
  - `expected_facts`（可选，数组）

示例：
```json
[
  {
    "case_id": "sample-1",
    "input_text": "我周五要提交预算表",
    "expected_facts": ["周五", "预算表"]
  }
]
```

## 3. 大数据集如何控制运行量？
推荐使用内置参数抽样运行：
- `sample_size`：本次运行取多少条。
- `start_index`：从第几条开始。

例如：
- 第一次跑前 100 条：`sample_size=100, start_index=0`
- 第二次跑下一批 100 条：`sample_size=100, start_index=100`

## 4. API
- `GET /api/datasets`：列出内置数据集及条目数。
- `POST /api/benchmark/run-dataset`：按 `sample_size/start_index` 运行指定内置数据集。
- `POST /api/benchmark/run-batch`：运行前端上传的 JSON 批量数据。

## 5. 建议
- 大数据集先跑小样本（例如 50~200）验证配置，再扩大量级。
- 真实线上数据建议先脱敏再放入 `backend/datasets/`。
