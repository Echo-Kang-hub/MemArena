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
- 输出健康性保护：当 LLM 返回 provider 错误文本（如 401/429）或本地占位文本时，不直接入库；会自动回退到启发式摘要，避免错误信息污染长期记忆。

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
- 熔断说明：当 `entity_extract_*` 通道首次出现 401/429 后，当前后端进程会对该实体 API 路由启用轻量熔断，后续同路由请求直接走本地回退，避免重复外呼与重复告警。

### EntityExtractor.mem0 系列补充
- 功能作用：面向“长期记忆事实化”，优先抽取可复用事实，不强调三元组结构。
- 当前实现修正：mem0 在处理阶段采用“双路抽取”策略（user + assistant 同轮抽取），并分别入库打上 `role:user` / `role:assistant`。
- 输入约束：
	- user 通道读取用户文本。
	- assistant 通道读取 `metadata.assistant_message`。
	- `mem0_user_facts / mem0_agent_facts / mem0_dual_facts` 在兼容层仍保留，但运行时统一走双路抽取。
- 检索与决策：先由引擎（常见是 `VectorEngine`）做相似检索，再由 Reflector（推荐 `Consolidator`）产出 `memory_decision`（keep/update/drop）。
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
- 角色定位：**冲突裁决器**（谁对谁错、旧值如何降级）。

### Consolidator
- 定义：相似度检测 + 融合式更新，减少冗余并处理属性过期。
- 适用：用户身份/属性发生阶段性变化（如“曾经是程序员，现在是摄影师”）。
- 纠错策略：可给出决策建议，但默认不替代冲突裁决写回。
- 对应引擎：`VectorEngine` / `RelationalEngine` / `GraphEngine`（反思器消费检索命中，理论上引擎无关）。
- LLM 方案：支持 `Heuristic / LLM / LLMWithFallback`。
- 角色定位：**去重融合器**（哪些应合并、保留、下沉）。
- 输出补充：支持 `memory_decision`（keep/update/drop）供上层策略消费。

### ConflictConsolidator
- 定义：组合反思器，串联 `Consolidator` 与 `ConflictResolver`，同时输出“融合决策 + 冲突裁决建议”。
- 适用：用户出现“我之前说错了/请更正”为代表的纠错场景，需要在同一轮里完成去重压缩与错误更正。
- 输出：
	- `memory_decision`（keep/update/drop，来自融合阶段）
	- `proposed_resolutions`（冲突裁决候选，来自冲突阶段）
- 对应引擎：`VectorEngine` / `RelationalEngine` / `GraphEngine`（三者均可适配）。
- LLM 方案：支持 `Heuristic / LLM / LLMWithFallback`。

### Consolidator 与 ConflictResolver 的区别
- 不重叠点：
	- Consolidator 关注“冗余与合并”（同义/近重复）。
	- ConflictResolver 关注“冲突与裁决”（互斥事实、版本胜出）。
- 实践建议：
	- 先 Consolidator 做压缩与候选决策，
	- 再 ConflictResolver 做最终冲突裁决与写回。
- 结论：Consolidator 不等同 ConflictResolver，二者是互补关系而非包含关系。

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

### Reflector 独立 LLM 路由
- 除 `retrieval.reflector_llm_mode` 外，Reflector 还支持独立 Provider 路由，配置键与 Summarizer/EntityExtractor 对齐：
	- `REFLECTOR_LLM_PROVIDER`
	- `REFLECTOR_API_BASE_URL`
	- `REFLECTOR_API_KEY`
	- `REFLECTOR_API_MODEL`
	- `REFLECTOR_OLLAMA_BASE_URL`
	- `REFLECTOR_OLLAMA_MODEL`
	- `REFLECTOR_LOCAL_MODEL_PATH`
- 前端可在配置面板中单独选择 `Reflector LLM Provider`，与 Chat/Judge/Summarizer/Entity 互不绑定。

## 6. STM/LTM 合并与展示约束
- 检索合并时会按内容去重（避免 STM 与 LTM 同文重复显示）。
- 前端 Markdown 报告会分开展示：
	- `Agent Real-time Memory (STM)`
	- `Agent Real-time Memory (LTM)`
- 说明：若内容完全相同，最终仅保留一条以减少噪声；优先保留 STM 命中顺序。 
- Prompt 注入分区：Assembler 会按角色输出独立分区，至少包含 `[MEMORY_USER]` 与 `[MEMORY_ASSISTANT]`（可选 `[MEMORY_OTHER]`）。
- 回归保障：已增加 API 端到端回归用例，断言最终 prompt 中必须出现 `[MEMORY_USER]` 与 `[MEMORY_ASSISTANT]`。
- 可视化展示：除 Markdown 报告外，Results Dashboard 也支持按 `STM/LTM × role(user/assistant/other)` 分组查看命中。

### STM 策略语义补充
- `short_term_memory_mode = none`：表示关闭 STM 检索与注入，仅走 LTM 路径；不等于关闭全部记忆。
- `isolate_sessions = true`（每个 case 独立会话，推荐）：批量评测中按 case 维度隔离 session，case 之间不串扰；不是按整份数据集共用一个会话。

## 7. Role-Split 记忆约束（User / Assistant）
- 所有 Memory Processor 产出的记忆块都应尽量按角色拆分，并附加角色标签：
	- `role:user`
	- `role:assistant`
- Engine 存储应保留 `metadata.role` 与 `metadata.tags`。
- 检索返回后，Assembler 在 prompt 中按角色分区注入（User Memory / Assistant Memory / Other Memory）。

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
