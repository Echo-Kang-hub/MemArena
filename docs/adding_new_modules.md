# 开发者指南：添加自定义模块

本文档演示如何在 MemArena 中新增一个模块实现，并接入前端下拉菜单与运行链路。

## 1. 新增 Processor
1. 在 `backend/app/implementations/processors/` 下创建新类。
2. 继承 `MemoryProcessor`，实现 `process()`。
3. 返回 `ProcessorOutput`，确保数据是标准 `MemoryChunk`。
4. 在 `backend/app/models/contracts.py` 的 `ProcessorType` 枚举中添加新值。
5. 在 `backend/app/registry.py` 的 `build_processor()` 注册映射。

示例：
```python
class KeywordProcessor(MemoryProcessor):
    def process(self, payload: RawConversationInput) -> ProcessorOutput:
        chunk = MemoryChunk(
            chunk_id=f"{payload.session_id}-kw-1",
            session_id=payload.session_id,
            content="关键词: ...",
            tags=["keyword"],
        )
        return ProcessorOutput(source=ProcessorType.keyword, chunks=[chunk])
```

## 2. 新增 Engine
1. 新建类并继承 `MemoryEngine`。
2. 必须同时实现 `save()` 与 `search()`。
3. 返回 `EngineSaveResult` / `EngineSearchResult`。
4. 更新 `EngineType` 枚举与 `build_engine()` 注册。

## 3. 新增 Assembler
1. 继承 `ContextAssembler`。
2. 输入 `AssembleRequest`，输出 `AssembleResult`。
3. 在返回值中附带 `preview_blocks`，便于前端结果可视化。
4. 更新 `AssemblerType` 与 `build_assembler()`。

## 4. 新增 Reflector（可选）
1. 继承 `MemoryReflector`，实现 async `reflect()`。
2. 反思逻辑应尽量无副作用。
3. 更新 `ReflectorType` 与 `build_reflector()`。

## 5. 新增 Evaluation Bench
1. 继承 `EvaluationBench`。
2. 实现 `evaluate()` 返回 `EvalResult`。
3. 若使用 LLM-as-a-Judge，建议输出可解释 `judge_rationale`。

## 6. 前端同步
1. 在 `frontend/src/types/index.ts` 扩展对应联合类型。
2. 在 `frontend/src/App.vue` 的下拉选项数组中加入新实现名称。
3. 若新增指标，在 `MetricBars.vue` 中加入渲染映射。

## 7. 兼容性建议
- 严格遵循 Pydantic 契约，避免返回自由结构。
- 新增枚举值后优先做一次端到端冒烟测试。
- 引擎若依赖外部服务（Milvus/Neo4j），建议在 docker-compose 增加健康检查。

## 8. 文档同步要求
- 新增或修改任一模块方案后，必须同步更新 `docs/memory_modules_catalog.md`。
- 若新增可配置参数（如检索参数、评估阈值），需同步更新 `README.md` 的 `.env 填写规则` 和前端配置说明。
