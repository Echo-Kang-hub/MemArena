# Memory 方案目录（可选实现总表）

本文件用于维护 MemArena 当前支持的所有可选模块方案。后续新增方案时，请同步补充本表。

## 1. Memory Processor（记忆预处理器）

### RawLogger
- 定义：原文全量入库，不做压缩。
- 优点：信息损失最小，回溯能力强。
- 缺点：上下文增长快，噪声较多。
- 适用：高保真审计、事实追踪。

### Summarizer
- 定义：把输入压缩成摘要后入库。
- 优点：节省存储与检索上下文。
- 缺点：可能丢失细节，受摘要质量影响。
- 适用：长会话、低成本检索。
- 可选方法：
	- `llm`：使用专用 Summarizer LLM 生成 3-5 条可检索摘要点。
	- `kmeans`：按句子 TF-IDF + KMeans 聚类选代表句摘要。
- 运行说明：当 `kmeans` 依赖不可用时，后端会自动降级为启发式摘要，流程不中断。

### EntityExtractor
- 定义：只提取实体（人名、地点、项目等）与结构化线索。
- 优点：检索速度快，结构清晰。
- 缺点：对隐含语义覆盖不完整。
- 适用：实体驱动问答、知识卡片。
- 可选方法：
	- `llm_triple`：LLM 三元组抽取。
	- `llm_attribute`：LLM 属性抽取。
	- `spacy_llm_triple`：spaCy 实体候选 + LLM 三元组抽取。
	- `spacy_llm_attribute`：spaCy 实体候选 + LLM 属性抽取。
- 引擎映射约束：
	- `*_triple` 必须使用 `GraphEngine`。
	- `*_attribute` 必须使用 `RelationalEngine`。
- 运行说明：当 spaCy 或模型不可用时，后端会自动降级为启发式实体候选。

## 2. Memory Engine（记忆黑盒引擎）

### VectorEngine（Chroma）
- 定义：向量化存储与语义检索，当前接入 Chroma 持久化。
- 可调参数：top_k、min_relevance、collection_name、similarity_strategy、keyword_rerank。
- 适用：语义相似检索。

### GraphEngine
- 定义：图结构记忆引擎（当前为内存占位实现）。
- 适用：关系推理、路径查询。
- 推荐搭配：`EntityExtractor` 的 `llm_triple` / `spacy_llm_triple`。

### RelationalEngine
- 定义：关系型记忆引擎（当前为内存占位实现）。
- 适用：结构化筛选、规则检索。
- 推荐搭配：`EntityExtractor` 的 `llm_attribute` / `spacy_llm_attribute`。

## 3. Context Assembler（上下文组装器）

### SystemInjector
- 定义：将记忆注入系统提示词上下文。
- 特点：简单直观，通用性强。

### XMLTagging
- 定义：用 XML 标签隔离记忆与用户输入。
- 特点：结构边界明确，便于模型遵循。

### TimelineRollover
- 定义：按时间线平铺记忆片段。
- 特点：对时序任务可读性更强。

## 4. Memory Reflector（异步反思器）

### None
- 定义：关闭反思旁路。

### GenerativeReflection
- 定义：自动生成高阶洞察建议。

### ConflictResolver
- 定义：检测潜在冲突并给出消解建议。

## 5. Evaluation Bench（评估台）

### LLMJudgeBench
- 定义：真实 LLM-as-a-Judge（支持 API 与 Ollama）。
- 输出：precision、faithfulness、info_loss、judge_rationale。
- 输出扩展：raw_judge_output（用于排查评估模型返回格式问题）。
- 说明：当外部模型不可用时，会回退到规则评估以保障流程可运行。

## 维护约定
- 新增任何方案后，必须同步更新本文件。
- 方案下线或行为变化后，必须同步修改“定义/优缺点/适用场景”。
