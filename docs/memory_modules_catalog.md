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
	- `mem0_user_facts`：mem0 风格用户事实提取（仅从用户文本抽取）。
	- `mem0_agent_facts`：mem0 风格助手事实提取（仅从 assistant_message 抽取）。
	- `mem0_dual_facts`：同时抽取 user + assistant 双通道事实。
- 引擎映射约束：
	- `*_triple` 必须使用 `GraphEngine`。
	- `*_attribute` 与 `mem0_*` 必须使用 `RelationalEngine`。
- 运行说明：当 spaCy 不可用时，后端会自动降级为启发式实体候选；当外部 LLM 不可用（如 401/429）或返回不可解析结果时，会回退为启发式三元组/属性抽取，避免结构化输出为空。

### EntityExtractor.mem0 系列补充
- 功能作用：面向“长期记忆事实化”，优先抽取可复用事实，不强调三元组结构。
- 输入约束：
	- `mem0_user_facts` 仅看用户文本。
	- `mem0_agent_facts` 仅看 `metadata.assistant_message`。
	- `mem0_dual_facts` 分别抽取并合并。
- 输出结构：统一落为属性型条目，便于 `RelationalEngine` 检索与后续审计。
- 优点：对偏好、计划、约束类事实更鲁棒。
- 缺点：关系路径推理能力弱于三元组模式。

## 2. Memory Engine（记忆黑盒引擎）

### VectorEngine（Chroma）
- 定义：向量化存储与语义检索，当前接入 Chroma 持久化。
- 可调参数：top_k、min_relevance、collection_name、similarity_strategy、keyword_rerank。
- 适用：语义相似检索。

### GraphEngine
- 定义：图结构记忆引擎（当前为内存占位实现）。
- 适用：关系推理、路径查询。
- 推荐搭配：`EntityExtractor` 的 `llm_triple` / `spacy_llm_triple`。
- 检索相关度（结构化评分）：
	- 评分维度：lexical（词项重叠）、entity（查询实体命中）、completeness（三元组字段完整性）、hint（写入时 score_hint）。
	- 默认权重：lexical=0.25, entity=0.40, completeness=0.30, hint=0.05。
	- 调参变量：`GRAPH_RELEVANCE_LEXICAL_WEIGHT`、`GRAPH_RELEVANCE_ENTITY_WEIGHT`、`GRAPH_RELEVANCE_COMPLETENESS_WEIGHT`、`GRAPH_RELEVANCE_HINT_WEIGHT`、`GRAPH_RELEVANCE_FALLBACK_LEXICAL_WEIGHT`、`GRAPH_RELEVANCE_FALLBACK_HINT_WEIGHT`。
	- 说明：后端会自动归一化权重；fallback 用于结构化字段缺失时回退到词法相关度。

### RelationalEngine
- 定义：关系型记忆引擎（当前为内存占位实现）。
- 适用：结构化筛选、规则检索。
- 推荐搭配：`EntityExtractor` 的 `llm_attribute` / `spacy_llm_attribute`。
- 检索相关度（结构化评分）：
	- 评分维度：lexical（词项重叠）、entity（主体/属性与查询命中）、completeness（属性键值完整性）、hint（写入时 score_hint）。
	- 默认权重：lexical=0.30, entity=0.35, completeness=0.25, hint=0.10。
	- 调参变量：`RELATIONAL_RELEVANCE_LEXICAL_WEIGHT`、`RELATIONAL_RELEVANCE_ENTITY_WEIGHT`、`RELATIONAL_RELEVANCE_COMPLETENESS_WEIGHT`、`RELATIONAL_RELEVANCE_HINT_WEIGHT`、`RELATIONAL_RELEVANCE_FALLBACK_LEXICAL_WEIGHT`、`RELATIONAL_RELEVANCE_FALLBACK_HINT_WEIGHT`。
	- 说明：后端会自动归一化权重；fallback 用于属性结构缺失时回退到词法相关度。

### 结构化相关度调参建议
- 平衡预设（默认）：保持现有默认值，适合大多数问答与卡片检索场景。
- 精确率优先：提高 `entity` 与 `completeness`，降低 `lexical`（减少“词相近但关系错”的命中）。
- 召回优先：提高 `lexical` 与 `fallback`（减少因抽取结构不完整导致的漏召回）。
- 排查建议：若相关度长期接近 0，优先检查抽取结果中是否含完整 `subject/predicate/object` 或 `entity/attribute/value` 字段。

## 3. Context Assembler（上下文组装器）

### SystemInjector
- 定义：将记忆注入系统提示词上下文。
- 特点：简单直观，通用性强。
- 置顶记忆（Pinned Memory）：支持。若 `metadata.pinned=true` 或标签含 `pinned/core`，会优先放在记忆块前部。

### XMLTagging
- 定义：用 XML 标签隔离记忆与用户输入。
- 特点：结构边界明确，便于模型遵循。

### TimelineRollover
- 定义：按时间线平铺记忆片段。
- 特点：对时序任务可读性更强。

### ReverseTimeline
- 定义：倒序时间线平铺（最近记忆优先）。
- 特点：强化 Recency 偏好，适合“最新状态优先”的场景。

### RankedPruning
- 定义：按相关度评分排序后，在 Token Budget 内动态截断低分片段。
- 特点：优先保证高相关记忆进入上下文，降低噪声干扰。
- 参数：`retrieval.max_context_tokens`（默认读取后端 `context_token_budget`）。

### ReasoningChain
- 定义：针对 `GraphEngine` 注入“推理路径 + 参考记忆”，而非仅注入最终事实。
- 特点：显式提供逻辑链条，提升复杂关系推理可解释性。
- 参数：`retrieval.reasoning_hops`（默认 1，范围 1~3）。
- 引擎要求：建议配合 `GraphEngine`；后端会调用多跳邻居扩展并在 metadata 中生成 `reasoning_chains`。
- 排序与压缩：优先保留“命中种子实体 + 低跳数”的链路，并去重后截断到 `GRAPH_REASONING_MAX_CHAINS`，避免链路噪声与冗余。
- 质量明细：在 metadata 中额外输出 `reasoning_chain_details`，包含 `hop`、`seed_touch`、`lexical_overlap`、`priority`，用于前端可解释展示与调参诊断。

## 4. Memory Reflector（异步反思器）

### None
- 定义：关闭反思旁路。

### GenerativeReflection
- 定义：自动生成高阶洞察建议。
- LLM 方案：支持 `Heuristic / LLM / LLMWithFallback`。

### ConflictResolver
- 定义：检测潜在冲突并给出消解建议。
- 适用：更正/纠错场景，关注错误事实的收敛清理效率。
- LLM 方案：支持 `Heuristic / LLM / LLMWithFallback`。
- 自动写回联动：可把 `proposed_resolutions` 写回控制块，用于检索阶段抑制过时值分数。

### Consolidator
- 定义：相似度检测 + 融合式更新 + 错误纠正，减少冗余并处理属性过期。
- 适用：用户身份/属性发生阶段性变化（如“曾经是程序员，现在是摄影师”）。
- 纠错策略：若用户明确指出之前错误，优先清理错误事实并保留纠正后事实。
- 对应引擎：`RelationalEngine` / `GraphEngine`。
- LLM 方案：支持 `Heuristic / LLM / LLMWithFallback`。

### DecayFilter
- 定义：基于 Ebbinghaus 遗忘曲线打分，按保留分衰减长尾记忆权重。
- 适用：长对话中的性能退化控制与检索噪声抑制。
- 对应引擎：`VectorEngine` / `RelationalEngine`。

### InsightLinker
- 定义：异步推理潜在实体连接（Link Prediction），补全缺失关系。
- 适用：知识图谱稀疏、知识孤岛场景。
- 对应引擎：`GraphEngine`。
- LLM 方案：支持 `Heuristic / LLM / LLMWithFallback`。

### AbstractionReflector
- 定义：将行为序列抽象为性格/偏好总结，形成高层用户画像。
- 适用：长期个性化与偏好建模。
- 对应引擎：`RelationalEngine`。
- LLM 方案：支持 `Heuristic / LLM / LLMWithFallback`。

### Reflector 全局 LLM 模式
- 配置项：`retrieval.reflector_llm_mode`
	- `Heuristic`：只走规则/启发式。
	- `LLM`：优先走 LLM。
	- `LLMWithFallback`：LLM 失败时自动回退启发式（默认推荐）。
- 前端已提供统一下拉选择。

## 6. STM/LTM 合并与展示约束
- 检索合并时会按内容去重（避免 STM 与 LTM 同文重复显示）。
- 前端 Markdown 报告会分开展示：
	- `Agent Real-time Memory (STM)`
	- `Agent Real-time Memory (LTM)`
- 说明：若内容完全相同，最终仅保留一条以减少噪声；优先保留 STM 命中顺序。 

## 5. Evaluation Bench（评估台）

### LLMJudgeBench
- 定义：真实 LLM-as-a-Judge（支持 API 与 Ollama）。
- 输出：precision、faithfulness、info_loss、judge_rationale。
- 输出扩展：raw_judge_output（用于排查评估模型返回格式问题）。
- 说明：当外部模型不可用时，会回退到规则评估以保障流程可运行。
- Reflector 相关指标：`convergence_speed`（收敛速度）。
- 指标含义：在用户更正后，系统预计需要几轮对话/反思周期才能将错误事实彻底洗掉；数值越小越好。
- Assembler 相关指标：`context_distraction`（上下文干扰度）。
- 指标含义：注入的检索上下文中，非相关记忆占比估计值；数值越低越好。

### ConvergenceSpeed 冒烟验证
- 推荐命令：`D:/AppDownload/miniconda3/envs/mema_env/python.exe backend/scripts/smoke_convergence_speed.py`
- 脚本特性：
	- 使用 FastAPI `TestClient` 进行 API 内部调用，不依赖外部 uvicorn 进程。
	- 使用 Python 原生 Unicode 字符串，避免 Windows 终端中文乱码导致的误判。
	- 同时输出：
		- `ConflictResolver` / `Consolidator` 的 API 回归结果；
		- 双冲突组确定性探针（期望 `convergence_speed=3.0`）。
	- 自动写入报告到 `backend/data/reports/`：
		- `convergence_smoke_YYYYMMDD_HHMMSS.json`
		- `convergence_smoke_YYYYMMDD_HHMMSS.csv`

### 批量导出（Reasoning 对比）
- 批量 CSV 会额外输出推理链质量列，便于不同 Assembler 横向比较：
	- `reasoning_chain_count`
	- `reasoning_avg_priority`
	- `reasoning_avg_hop`
	- `reasoning_seed_touch_ratio`

## 维护约定
- 新增任何方案后，必须同步更新本文件。
- 方案下线或行为变化后，必须同步修改“定义/优缺点/适用场景”。
