# 产品说明书：DeepScholar Agent (科研深读助手)

## 1. 项目概述 (Executive Summary)

### 1.1 产品定义
**DeepScholar** 是一个基于 **Agentic Workflow（代理工作流）** 的科研辅助系统。不同于传统的“切片+向量检索（RAG）”模式，DeepScholar 模拟人类专家的阅读习惯，通过“**渐进式披露**”机制，自主决定阅读策略（先看大纲，再查细节），并在**思维工作台（Workspace）** 中积累证据，最终在限定的参考论文范围内（如 20 篇核心文献）进行深度头脑风暴和创新点挖掘。

### 1.2 核心价值
*   **摆脱碎片化：** 解决传统 RAG 丢失跨段落逻辑和宏观语境的问题。
*   **深度推理：** 通过维护全局状态（State），支持跨多篇论文的矛盾检测、方法迁移和空白点挖掘。
*   **精准可控：** 仅依赖用户提供的限定高质量文献，杜绝幻觉，确保 Context 纯净。

---

## 2. 技术架构 (Technical Architecture)

本项目采用全 **Python** 技术栈，强调本地化处理与高性能交互。

### 2.1 核心技术栈
*   **开发语言：** Python 3.10+
*   **大模型基座：** **使用英伟达NIM平台模型，调用示范如下：
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        client = ChatNVIDIA(
        model="qwen/qwen3.5-397b-a17b",
        api_key="$NVIDIA_API_KEY",
        temperature=0.6,
        top_p=0.95,
        max_completion_tokens=16384,
        )。
*   **Agent 编排：** **LangGraph**
    *   *理由：* 支持循环图结构（Cyclic Graph），完美适配“阅读-思考-再阅读”的迭代流程；支持持久化状态管理（Checkpointer）。
*   **前端交互：** **Chainlit**
    *   *理由：* Python 原生，专为 LLM 设计。支持“思维链”可视化（Step UI）和侧边栏（Side View），适合展示“思维工作台”。
*   **后端服务：** **FastAPI** (可选，通过 LangServe 暴露 API)。
*   **文档解析：** **MinerU (Magic-PDF)**
    *   *理由：* 优秀的 PDF 转 Markdown 能力，支持公式还原。
*   **存储层：** 本地文件系统 (Markdown/JSON) + **BM25** (内存级关键词索引)，无需重型向量数据库。

### 2.2 数据流向
1.  **Ingestion:** PDF $\rightarrow$ MinerU $\rightarrow$ Markdown $\rightarrow$ 结构化切分 $\rightarrow$ 索引生成。
2.  **Runtime:** User Query $\rightarrow$ Agent Brain (LangGraph) $\rightarrow$ Tool Calls (Read/Search) $\rightarrow$ Update Workspace $\rightarrow$ Final Answer.

---

## 3. 功能详解 (Feature Specifications)

### 3.1 模块一：智能预处理 (The Ingestion Pipeline)
*此模块负责将非结构化数据转化为高语义密度的知识库。*

*   **F1.1 高保真解析：** 使用 MinerU 将 PDF 转为 Markdown，保留公式。
*   **F1.2 语义分块与清洗：**
    *   识别 Markdown 的一级标题（H1），若章节过长（>2k tokens），自动递归切分至 H2/H3。
    *   去除参考文献列表（References），建立引用锚点链接。
*   **F1.3 图表语义增强 (Visual Understanding)：**
    *   自动提取论文中的 Figure 和 Table。
    *   调用 VLM (Vision Language Model) 为每个图表生成详细描述（Caption + Insight），插入 Markdown 对应位置。
*   **F1.4 索引卡片生成 (Indexing)：**
    *   对每个 Section 生成 JSON 索引：包含 `Summary` (摘要), `Tags` (关键实体), `Claims` (核心贡献), `Hypothetical_Questions` (该节解决的问题)。

### 3.2 模块二：思维工作台 (The Thinking Workspace)
*这是系统的“大脑”缓存区，区别于聊天记录。*

*   **F2.1 全局状态管理：** 在 System Prompt 中维护一个 `<workspace>` 区域。
*   **F2.2 动态更新：** Agent 在阅读过程中，将发现的“证据”、“矛盾点”、“灵感片段”实时写入 Workspace。
*   **F2.3 侧边栏展示：** 前端 (Chainlit) 实时渲染 Workspace 的内容，用户可以看到 Agent 的“笔记本”在逐渐变厚。

### 3.3 模块三：Agent 核心逻辑 (The Researcher)
*基于 LangGraph 定义的状态机。*

*   **F3.1 工具集 (Tools)：**
    *   `get_library_structure()`: 获取所有论文的标题 + 顶层大纲。
    *   `read_section_content(paper_id, section_title)`: 精确读取某篇论文的某章节全文。
    *   `keyword_search(query)`: 基于 BM25 的全文关键词检索（补漏用）。
    *   `update_notes(content)`: 写入思维工作台。
*   **F3.2 思考模式 (Reasoning Patterns)：**
    *   **Map-Reduce:** 先看大纲 (Map)，筛选 3-5 篇最相关论文，再深入读取 (Reduce)。
    *   **Conflict Check:** 专门寻找两篇论文实验设置或结论的差异。

---

## 4. 交互设计 (UI/UX via Chainlit)

### 4.1 界面布局
*   **主窗口 (Chat Interface):**
    *   用户输入框。
    *   流式对话输出。
    *   **Step 展开:** 显示 "Reading Paper A...", "Extracting Method..." 等中间步骤，让用户感知 Agent 的工作量。
*   **右侧边栏 (Side View - "Research Notes"):**
    *   这是一个 Markdown 渲染区域。
    *   显示 Agent 当前的 **Workspace** 内容。
    *   包含：已读论文列表、关键引用摘录、初步生成的 Idea 草稿。

### 4.2 典型对话流程
1.  **用户:** “结合这几篇论文，如果我们想改进 Transformer 的位置编码，有什么新思路？”
2.  **系统:** (显示 "Thinking...")
    *   *Action:* 扫描所有论文大纲，找到涉及 Positional Encoding 的章节。
    *   *Action:* 选中 Paper A (RoPE), Paper B (ALiBi)。
    *   *Action:* 读取 Paper A 的 Method 节，读取 Paper B 的 Experiments 节。
    *   *Update Side View:* 在右侧记录 A 和 B 的优缺点对比。
3.  **系统:** (输出回答) “通过对比 A 和 B，发现 A 在长外推时性能下降，而 B 牺牲了部分精度。一个可能的 Idea 是结合 A 的旋转机制和 B 的线性偏置……”

---

## 5. 开发路线图 (Implementation Roadmap)

### Phase 1: 基础构建 (MVP) - *预计耗时: 1周*
*   [Backend] 搭建 Python 环境，集成 MinerU。
*   [Data] 实现 `PDF -> Markdown` 的转换脚本。
*   [Data] 实现简单的 H1 标题切分。
*   [Agent] 使用 LangGraph 构建最简单的 "Router": 用户问 -> Agent 选论文 -> 读全文 -> 回答。
*   [UI] 跑通 Chainlit Hello World。

### Phase 2: 渐进式阅读与工作台 - *预计耗时: 2周*
*   [Agent] 升级 LangGraph 逻辑：实现 `Outline -> Select -> Read Section` 的循环。
*   [Agent] 引入 `AgentState` 中的 `workspace` 字段。
*   [UI] 在 Chainlit 中实现 Side View，实时展示 `workspace` 内容。
*   [Data] 优化索引：为每个 Section 生成 Tags 和 Summaries。

### Phase 3: 深度增强 - *预计耗时: 2周*
*   [Data] 集成 VLM (GPT-4o/Claude Vision)，实现图表转文字描述，插入 Markdown。
*   [Search] 引入 BM25 实现关键词补漏检索。
*   [Prompt] 调优 System Prompt，增加“寻找矛盾”、“方法迁移”等特定指令。

---

## 6. 文件结构建议 (Project Structure)

```text
deep_scholar/
├── data/
│   ├── raw_pdfs/              # 原始 PDF
│   ├── processed/             # 处理后的数据
│   │   ├── paper_id_01/
│   │   │   ├── full.md        # MinerU 结果
│   │   │   ├── index.json     # 结构化索引 (Summary/Tags)
│   │   │   └── sections/      # 切分后的章节 MD 文件
├── src/
│   ├── ingestion/
│   │   ├── parser.py          # MinerU 调用
│   │   ├── splitter.py        # 层级切分逻辑
│   │   └── indexer.py         # LLM 摘要与 Tag 生成
│   ├── agent/
│   │   ├── graph.py           # LangGraph 定义 (核心逻辑)
│   │   ├── tools.py           # read_section, search 等工具函数
│   │   └── prompts.py         # System Prompts
│   └── app.py                 # Chainlit 启动入口
├── requirements.txt
└── .env                       # API Keys
```

---

## 7. 风险评估与对策

1.  **Token 消耗过大：**
    *   *对策：* 严格限制 Agent 单次 Loop 读取的章节数量（例如最多 3 个章节）；强制先看 Summary 再看 Full Text。
2.  **MinerU 解析失败：**
    *   *对策：* 增加回退机制，如果 MinerU 失败，回退到 PyMuPDF 提取纯文本（虽然丢失格式但保留内容）。
3.  **响应延迟：**
    *   *对策：* 利用 Chainlit 的流式输出；在 Agent 思考时，UI 必须展示动态进度条，缓解用户焦虑。