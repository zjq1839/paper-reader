# DeepScholar 记忆与上下文系统升级修改需求说明书 (MRD)

## 1. 项目概述与升级目标

当前系统的上下文窗口在处理深度的多模态文献（如涉及复杂网络架构图、长篇数学推导）时，易触发“90分钟悬崖”式的上下文截断与幻觉。本次升级旨在为 Agent 引入结构化的功能域记忆系统与混合检索基底，彻底解决长篇文献阅读中的事实遗忘、逻辑碎片化以及不同文献间结论冲突的问题。

## 2. 核心架构与技术栈变更

* **向量库迁移：** 废弃或降级原有的纯向量数据库依赖，在本地引入基于 SQLite 的混合检索库（结合 FTS5/BM25 全文检索与本地向量扩展）。
* **状态管理升级：** 深度利用 LangGraph 的图状态流转特性，将单一的 `chat_history` 升级为包含多维度记忆体（工作记忆、情景记忆、语义记忆）的复杂状态对象 (State Schema)。
* **依赖库新增 (`requirements.txt`)：** 预计新增 `rank_bm25` (或直接依赖 SQLite FTS)、本地轻量级向量方案（如 `sqlite-vec` 或 `faiss-cpu`）、以及用于计算时间衰减的工具库。

## 3. 具体模块修改需求细则

### 3.1 核心Agent逻辑与状态管理 (`src/agent/graph.py` & `src/agent/lc_agent.py`)

**需求描述：** 重构 Agent 的运转流与上下文维持机制，实现记忆的分层流转与静默压缩。

* **状态定义重构 (State Schema Update)：**
* 修改 LangGraph 的 State 定义，新增字段：`working_workspace` (当前研究上下文)、`semantic_directives` (全局不变指令)、`episodic_events` (历史探究轨迹)。


* **引入压缩代理节点 (Compaction Node)：**
* 在图中新增一个监控节点。当 `messages` 列表的 Token 数量达到设定阈值（如预设 Token 上限的 80%）时，拦截主控 Agent 的执行，将历史对话路由至 `Compaction Node`。
* 该节点负责执行**预压缩记忆刷新 (Pre-Compaction Flush)**，提取高信息密度的核心结论持久化到本地硬盘，并用简短的摘要对象替换长对话历史。


* **上下文硬注入机制：**
* 在图的入口节点 `START` 处，硬编码读取根目录下的核心约束文件（如 `DEEPSCHOLAR_CORE.md`，用于存储针对不可变的先验知识与工作流范式），确保其以最高优先级注入每一轮推理。



### 3.2 混合检索与数据注入基底 (`src/ingestion/indexer.py` & `src/ingestion/pipeline.py`)

**需求描述：** 改造文献的解析与索引入库流程，实现精准的双路召回融合。

* **双重索引构建 (`indexer.py`)：**
* 将解析后的文献分块（Chunks）同时写入两套索引库：一套是保留原汁原味专有名词的 BM25 倒排索引；另一套是基于本地嵌入模型生成的向量索引。
* 针对特定字母组合与连字符构成的专有名词，需优化分词器 (Tokenizer) 规则，避免关键检索词被错误切割。


* **加权融合召回引擎 (Hybrid Retriever)：**
* 在检索管道中新增一个融合函数 (Fusion Function)。当 Agent 触发查询时，分别获取向量相似度得分（余弦距离）和文本匹配得分（BM25）。
* 将两组分数进行标准化处理后执行加权融合（默认权重分配：70% 向量语义 + 30% 词法匹配），按融合得分排序返回 Top-K 结果。



### 3.3 提示词模板与冲突调解 (`src/agent/prompts.py`)

**需求描述：** 增加用于记忆调取、信息折叠以及文献观点冲突处理的专用 Prompt。

* **新增动态简报模板 (Dynamic Briefing Template)：**
* 设计具备严格行数限制的系统提示词模板，包含 `[Current Focus]`, `[Known Constraints]`, `[Key Findings]` 三个占位符。若内容超出预算，模板需自动折叠旧信息，并提示大模型：“存在隐藏的历史文献笔记，请使用 `search_episodic_memory` 工具查阅”。


* **新增语义冲突调解模板 (LLM-Mediated Reconciliation Prompt)：**
* 当检索出的多篇文献对同一技术方案（如小样本场景下的特征融合效率）存在矛盾描述时，不得直接将原始文本拼接喂给主 Agent。
* 需在此处新增一个专用的“调解 Prompt”，要求 LLM 作为第三方审查者，分析两篇文献的实验条件差异，并输出包含边界条件的合并结论。



### 3.4 工具链扩展 (`src/agent/tools.py`)

**需求描述：** 为主控 Agent 提供主动操作长期记忆的接口。

* 新增 `search_episodic_memory(query, time_range)`: 允许 Agent 检索带有时间戳的历史实验推演记录。内部实现需加入艾宾浩斯时间衰减因子，对远古记录进行降权。
* 新增 `update_semantic_knowledge(topic, conclusion)`: 允许 Agent 在确认了某篇高价值文献的关键突破后，主动将提炼出的研究模式写入全局 Tier 1 知识库。

## 4. 数据流向与执行时序示例

1. **输入阶段：** 用户提问关于某篇新摄入文献中关于视觉模型微调的问题。
2. **上下文装载：** 系统优先读取 `DEEPSCHOLAR_CORE.md` (Tier 1 语义记忆)，合并当前的工作区简报。
3. **检索阶段 (`indexer.py`)：** 解析查询，并行触发向量库与 BM25 检索，加权融合后返回核心文献块。
4. **冲突检测阶段：** 验证新抽取的文献块与 `working_workspace` 中的已有结论是否存在中度冲突（Jaccard/余弦相似度阈值判定）。若冲突，先通过调解 Prompt 运行一次子查询。
5. **推理与生成 (`graph.py`)：** 主控 Agent 基于整理好的完美无冲突上下文生成深度解答。
6. **维护阶段：** 若该轮交互导致 Token 触及警戒线，图状态流转至 `Compaction Node`，执行记忆提炼与对话框瘦身。

## 5. 实施阶段建议

* **Phase 1 (底层改造)：** 优先在 `requirements.txt` 和 `src/ingestion` 中完成 SQLite 与双路召回逻辑的替换。测试独立检索专有名词的命中率。
* **Phase 2 (大脑升级)：** 修改 `src/agent/graph.py` 中的 LangGraph 状态机，挂载工作记忆与情景记忆的读取逻辑。
* **Phase 3 (机制完善)：** 实装 Token 监控、预压缩刷新节点以及语义冲突调解机制。进行多篇连贯文献的压力测试，观察“90分钟悬崖”是否被消除。