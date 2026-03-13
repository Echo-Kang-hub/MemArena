# MemArena 评估标准（Evaluation Criteria）

## 1. 总体原则
- 评估必须覆盖检索质量、回答质量、一致性与安全性，不只看单一分数。
- 保留原有指标（Precision / Faithfulness / InfoLoss）作为检索与事实覆盖的基础盘。
- 引入业内通用指标（Recall@K、QA Accuracy/F1、Consistency、Rejection、ROUGE/BLEU/FactScore）形成全景评测。

## 2. 综合指标矩阵（建议主面板展示）

### 2.1 检索层（Retrieval）
- Recall@K：在前 K 条检索结果中命中关键事实的比例，越高越好。
- Precision：检索结果中有效信息占比，越高越好。
- Faithfulness：期望事实覆盖率，越高越好。
- InfoLoss：关键信息丢失率，越低越好。

### 2.2 回答层（Answer Quality）
- QA Accuracy：回答是否正确命中目标事实（可按样本级 0/1 统计）。
- QA F1：面向事实片段的精确率与召回率平衡，更适合部分命中场景。

### 2.3 一致性与安全层（Reliability & Safety）
- Consistency Score：多轮对话前后是否自洽、是否自我矛盾。
- Rejection Rate：模型实际拒答行为比例（该字段只反映“是否拒答”，不直接代表拒答是否正确）。
- Rejection@Unknown：仅在未知问题子集统计的拒答正确率（更贴近安全性目标）。

### 2.4 生成式回忆层（Generation）
- ROUGE：对摘要或事件回顾的 n-gram 重叠评估。
- BLEU：更偏机器翻译式的文本重叠评估，可作为补充。
- FactScore：生成文本中事实陈述的可验证正确率（推荐重点关注）。

## 3. 原有指标的价值与定位
- Precision / Faithfulness / InfoLoss 仍然有高价值：
- 它们直接反映“记忆检索是否找对、找全、丢没丢”。
- 在系统迭代初期，它们比复杂语义指标更稳定、可解释、可快速回归。
- 建议将其保留为基础指标，并与新指标联合判定。

## 4. 派生与稳定性指标（建议继续保留）
- F1 (P&R Balance)：Precision 与 Faithfulness 的调和均值。
- Retention：1 - InfoLoss，保留信息比例。
- Hallucination Risk：1 - Precision，噪声或幻觉风险近似值。
- Pass Rate：达阈值样本占比（默认 P>=0.8, F>=0.8, L<=0.2）。
- Worst-case F1：最差样本下限表现。
- Std(P/F/L)：批量指标标准差，衡量稳定性。

## 5. 判定建议（可作为默认门槛）
- 通过：Pass Rate >= 85%，Worst-case F1 >= 0.60，Recall@K 达到业务阈值。
- 可用但需优化：Pass Rate 在 60%~85%，或波动较大（Std > 0.15）。
- 高风险：Pass Rate < 60%，或 Consistency 明显偏低，或 Rejection Rate 异常。

## 6. 当前自动化能力与落地建议
- 当前已自动化：Precision、Faithfulness、InfoLoss、Recall@K、QA Accuracy/F1。
- 当前已自动化（第一版规则法）：Consistency Score、Rejection Rate、Rejection@Unknown。
- 建议半自动（先人工抽样，后自动化）：FactScore。
- 任务定制指标：ROUGE/BLEU 仅在摘要/回忆生成任务启用，避免误用。

### 6.1 评分方法说明
- LLM Judge：Precision、Faithfulness、InfoLoss。
- 规则法：Recall@K、QA Accuracy/F1、Consistency、Rejection、Rejection@Unknown。
- 建议在报告中同时展示“Judge 分”与“规则分”，避免单一评估偏差。
- Rejection@Unknown 即“未知问题子集上的拒答正确率”，用于补齐 Rejection Rate 的语义缺口。
- 建议同时展示 Unknown 样本占比（Unknown Ratio），避免在未知样本过少时误读 Rejection@Unknown。

## 7. 指标代表性边界
- 该指标组主要评价“记忆检索与事实覆盖质量”，不是通用模型总分。
- expected_facts 标注质量决定上限；漏标会导致虚高，错标会导致虚低。
- LLM-as-a-Judge 存在偏差，建议抽样人工复核并固定 judge 提示词版本。
- 实际评估应同时看均值、波动、下限、原始评审理由和失败样本。

## 8. 高挑战测试设计
- 干扰实体：同音/近形地名、人名、项目名混入。
- 时序冲突：多轮修改、撤销、改期，验证最新事实优先。
- 长上下文噪声：20~50 条历史中隐藏关键事实。
- 条件约束：否定句、条件句、多重约束句。
- 对抗样本：诱导性错误先验或伪事实注入。
- 多语言混合：中英混写、缩写、口语、错别字。

## 9. 数据集分层建议
- smoke：链路冒烟，小规模快速检查。
- regression：固定回归集，保障版本稳定。
- hard：高难样本集，拉开模型/策略差异。
