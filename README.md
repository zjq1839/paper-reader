# DeepScholar Agent（科研深读助手）

DeepScholar 是一个面向论文精读与研究灵感生成的 AI 研究助手：先将论文（PDF 或 PDF URL）离线处理为可检索的结构化 Markdown 分段库，再通过 Agent 工具调用按需阅读章节并持续维护一个“Research Workspace”（侧边栏可见的研究笔记区）。

## 项目背景与目标

- 背景：传统“切片 + 向量检索”的 RAG 容易丢失跨段落的宏观语境与逻辑链路，难以模拟专家式的“先看结构、再追细节”的阅读方式。
- 目标：以“渐进式披露（Progressive Disclosure）”为核心策略，让 Agent 先浏览论文库大纲，再按需读取章节内容，最终输出可追溯到原文段落的综合结论，并把关键证据/灵感沉淀到 Workspace。

## 上下文窗口（Context Window）策略

- 渐进式披露：优先 `hybrid_search` 获取短证据片段；仅在需要时分页读取章节内容（避免一次性加载超长 Markdown）。
- Token 阈值触发压缩：基于消息内容长度估算 Token 使用率，超过 `MAX_CONTEXT_TOKENS` 的 95% 时自动触发“摘要压缩”。
- 摘要式续写：压缩节点会把旧历史融合进 `summary`，后续对话用「summary + 最近消息」维持推理链不断裂。
- 状态感知：系统提示词会注入已读章节列表（READ HISTORY），降低重复读取同一章节的概率。

## 技术栈

- 语言：Python
- UI：Chainlit（聊天 UI + 侧边栏 Workspace + Agent Trace）
- Agent：LangGraph + LangChain + NVIDIA NIM（`langchain-nvidia-ai-endpoints` 的 `ChatNVIDIA`）
- 文档解析：MinerU（`magic-pdf`，本地或 API）+ PyMuPDF（失败回退/本地解析）
- 配置：python-dotenv（加载 `.env`）
- 检索：SQLite FTS5（BM25）+ 可选 FAISS 向量检索（NVIDIA Embedding / bge-m3）

## 系统架构（模块划分与数据流）

```mermaid
flowchart TD
  U[用户] -->|提问/对话| CL[Chainlit UI\nsrc/app.py]
  CL -->|messages(history)| A[Agent\nsrc/agent/lc_agent.py]
  A -->|tool call| T1[get_library_structure\nsrc/agent/tools.py]
  A -->|tool call| T2[hybrid_search\nsrc/agent/tools.py]
  A -->|tool call| T3[read_section_content(分页)\nsrc/agent/tools.py]
  A -->|tool call| T4[update_workspace\nsrc/agent/tools.py]
  A -->|tool call| T5[report_status\nsrc/agent/tools.py]
  T1 --> FS[(data/processed\nindex.json)]
  T2 --> FS[(data/hybrid_index.db + faiss_index)]
  T3 --> FS
  T4 --> CL
  T5 --> CL
  CL -->|trace events| TR[(data/memory/traces\n*.jsonl)]

  subgraph ING[离线摄取（Ingestion）]
    PDF[PDF / PDF URL\n(data/raw_pdfs)] --> P[PDFParser\nsrc/ingestion/parser.py]
    P --> M[Markdown]
    M --> S[MarkdownSplitter\nsrc/ingestion/splitter.py]
    S --> OUT[data/processed/<paper_id>\nindex.json + sections/*.md]
  end
```

### 核心模块与依赖关系

- 摄取流水线（`src/ingestion/*`）
  - [`pipeline.py`](file:///e:/work/paper-reader/src/ingestion/pipeline.py)：遍历 `data/raw_pdfs/`，对 PDF 文件与 URL 列表进行处理，输出到 `data/processed/`。
  - [`parser.py`](file:///e:/work/paper-reader/src/ingestion/parser.py)：优先使用 MinerU（本地 `magic-pdf` 或 MinerU API），失败回退到 PyMuPDF（仅支持本地 PDF）。
  - [`splitter.py`](file:///e:/work/paper-reader/src/ingestion/splitter.py)：按一级标题（H1）切分 Markdown 为章节分段。
- Agent 与工具（`src/agent/*`）
  - [`lc_agent.py`](file:///e:/work/paper-reader/src/agent/lc_agent.py)：LangGraph 工作流入口（对外暴露 `agent`）。
  - [`prompts.py`](file:///e:/work/paper-reader/src/agent/prompts.py)：约束 Agent 的“先看大纲再读章节”“先 `report_status` 再调用其它工具”等策略。
  - [`tools.py`](file:///e:/work/paper-reader/src/agent/tools.py)：论文结构、混合检索（SQLite FTS + 可选向量）、章节分页读取，并通过 `WORKSPACE_UPDATE::` 协议与 UI 同步侧边栏笔记。
  - [`graph.py`](file:///e:/work/paper-reader/src/agent/graph.py)：上下文窗口策略与生命周期管理（Token 阈值触发摘要压缩、summary 注入、READ HISTORY 注入）。
- UI 入口（`src/app.py`）
  - [`app.py`](file:///e:/work/paper-reader/src/app.py)：Chainlit 事件处理（chat_start/on_message）、历史消息管理、侧边栏 Workspace 渲染、LLM/工具轨迹 Step 展示，并将事件落盘到 `data/memory/traces/*.jsonl`。

## 项目目录结构

```text
e:\work\paper-reader
├── .env.example
├── .chainlit/
│   └── config.toml
├── data/
│   ├── raw_pdfs/                  # 原始 PDF 或 URL 列表（.txt/.urls）
│   └── processed/                 # 摄取后：每篇论文一个目录
│       └── <paper_id>/
│           ├── index.json          # 标题 + sections 元数据
│           └── sections/
│               └── section_*.md    # 分段后的章节内容
│   └── memory/                     # 运行期记忆/轨迹（启动后自动创建）
│       └── traces/                 # 每轮对话的事件轨迹（jsonl）
├── src/
│   ├── app.py
│   ├── agent/
│   │   ├── lc_agent.py
│   │   ├── prompts.py
│   │   ├── tools.py
│   │   └── graph.py               # LangGraph 工作流定义
│   └── ingestion/
│       ├── pipeline.py
│       ├── parser.py
│       ├── splitter.py
│       └── indexer.py             # HybridIndexer：SQLite FTS + 可选 FAISS
├── transformers/
│   └── __init__.py                # 轻量 shim（避免某些依赖强制引入 transformers）
├── requirements.txt
└── tests/
    └── test_agent.py
```

## 环境准备

- Python：建议 3.10+（至少需满足 Chainlit 与 LangChain 的依赖要求）
- 可选：Conda / venv
- 必需：NVIDIA NIM API Key（用于 `ChatNVIDIA`）
- 可选：MinerU Token（解析 PDF URL 或使用 MinerU API 时需要）

## 安装步骤（Windows / PowerShell）

```powershell
cd e:\work\paper-reader
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果你使用 Anaconda 且 Python 在 `E:\anaconda`、环境名为 `my`，也可以直接：

```powershell
cd e:\work\paper-reader
E:\anaconda\envs\my\python.exe -m pip install -r requirements.txt
```

## 关键配置项说明

### 1) `.env`（推荐）

把 `.env.example` 复制为 `.env`，并填入真实值：

```powershell
cd e:\work\paper-reader
Copy-Item .env.example .env
```

`.env.example` 中已有的占位符保持原样如下（按需填写）：

```dotenv
NVIDIA_API_KEY=your_nvidia_api_key_here
MINERU_API_TOKEN=your_mineru_api_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 2) 运行时环境变量（可选覆盖）

Agent 相关（见 [`graph.py`](file:///e:/work/paper-reader/src/agent/graph.py)）：

- `NVIDIA_API_KEY`：必填；未设置会直接报错
- `NVIDIA_MODEL`：默认 `qwen/qwen3.5-397b-a17b`
- `NVIDIA_MAX_TOKENS`：单次回答的最大输出 token（completion tokens），默认 `4096`
- `MAX_CONTEXT_TOKENS`：上下文窗口估算上限（用于触发摘要压缩），默认 `32000`

混合检索相关（见 [`tools.py`](file:///e:/work/paper-reader/src/agent/tools.py) 与 [`indexer.py`](file:///e:/work/paper-reader/src/ingestion/indexer.py)）：

- `HYBRID_INDEX_DB_PATH`：SQLite FTS 索引路径，默认 `data/hybrid_index.db`
- `HYBRID_VECTOR_DB_PATH`：向量索引目录，默认 `data/faiss_index`
- `HYBRID_VECTOR_ENABLED`：是否启用向量检索，默认 `1`（可设为 `0` 仅用 FTS）
- `HYBRID_SNIPPET_MAX_CHARS`：`hybrid_search` 返回片段最大字符数，默认 `1400`
- `NVIDIA_BASE_URL`：Embedding/推理 API Base URL，默认 `https://integrate.api.nvidia.com/v1`
- `NVIDIA_EMBED_MODEL`：Embedding 模型名，默认 `baai/bge-m3`

MinerU 相关（见 [`parser.py`](file:///e:/work/paper-reader/src/ingestion/parser.py)）：

- `MINERU_API_TOKEN` / `MINERU_TOKEN`：解析 PDF URL 或调用 MinerU API 时需要
- `MINERU_API_BASE_URL`：默认 `https://mineru.net/api/v4`
- `MINERU_MODEL_VERSION`：默认 `vlm`

PowerShell 临时设置示例（会话级）：

```powershell
$env:NVIDIA_API_KEY="your_nvidia_api_key_here"
$env:MINERU_API_TOKEN="your_mineru_api_token_here"
```

## 本地开发指南

### 1) 摄取论文（生成可读库）

方式 A：放入本地 PDF 到 `data/raw_pdfs/` 后运行：

```powershell
cd e:\work\paper-reader
.\.venv\Scripts\python.exe -m src.ingestion.pipeline
```

方式 B：在 `data/raw_pdfs/` 放入 `.txt` 或 `.urls` 文件，每行一个 PDF URL（示例文件：`data/raw_pdfs/papers.txt`），再运行同一条命令。

输出会写入 `data/processed/<paper_id>/index.json` 与 `data/processed/<paper_id>/sections/section_*.md`。

### 2) 启动交互 UI（Chainlit）

开发模式（自动热重载）：

```powershell
cd e:\work\paper-reader
.\.venv\Scripts\python.exe -m chainlit run src/app.py -w
```

启动后：

- 右侧边栏包含 **Research Workspace** 与 **Agent Trace**（LLM/工具的时间线与耗时）
- 每轮对话的轨迹会追加写入 `data/memory/traces/<trace_session_id>.jsonl`

## 生产部署指南

最小化方式：直接以非热重载模式启动，并绑定监听地址（适合配合反向代理）：

```powershell
cd e:\work\paper-reader
.\.venv\Scripts\python.exe -m chainlit run src/app.py --host 0.0.0.0 --port 8000
```

说明：

- `.chainlit/config.toml` 中 `allow_origins = ["*"]` 适合开发调试；生产建议收敛到实际域名白名单。
- 当前仓库未实现 FastAPI 路由/HTTP API（尽管依赖已列出）；若需要对外提供 HTTP API，建议新增一个 FastAPI `app` 并复用 `src/agent/*` 与 `src/ingestion/*`。

## 单元测试与集成测试

当前测试基于 `unittest`（见 [`tests/test_agent.py`](file:///e:/work/paper-reader/tests/test_agent.py)），覆盖：

- `get_library_structure`：能从 `data/processed/*/index.json` 读取结构
- `read_section_content`：能根据 section title 读取对应 Markdown 文件（长章节支持分页读取，默认读取第 1 页）

运行方式（Windows / PowerShell）：

```powershell
cd e:\work\paper-reader
.\.venv\Scripts\python.exe -m unittest -v
```

或（显式 discover）：

```powershell
cd e:\work\paper-reader
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## API / 接口使用示例

### 1) 作为 Python 模块调用摄取能力

```python
from src.ingestion.parser import PDFParser
from src.ingestion.splitter import MarkdownSplitter

parser = PDFParser()
md = parser.parse("e:\\work\\paper-reader\\data\\raw_pdfs\\example.pdf")

splitter = MarkdownSplitter()
sections = splitter.split(md)
print(len(sections), sections[0]["title"])
```

### 2) 直接调用工具（与 Agent 内部一致）

```python
from src.agent.tools import get_library_structure, read_section_content

library = get_library_structure.invoke({})
print(library)

content = read_section_content.invoke({"paper_id": "paper_a", "section_title": "Abstract"})
print(content)

# 长章节分页读取（第 2 页）
page2 = read_section_content.invoke({"paper_id": "paper_a", "section_title": "Method", "page": 2})
print(page2)
```

## 贡献规范

### 分支策略

- `main`：可运行、可发布的稳定分支
- `feat/<scope>-<short>`：新功能
- `fix/<scope>-<short>`：缺陷修复
- `chore/<scope>-<short>`：工程化/依赖/杂项

### 代码风格

当前仓库未内置强制格式化/静态检查配置。建议本地统一采用：

```powershell
cd e:\work\paper-reader
.\.venv\Scripts\python.exe -m pip install -U ruff black
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m black --check .
```

### 提交信息格式（Conventional Commits）

- `feat: ...` 新功能
- `fix: ...` 修复
- `docs: ...` 文档
- `refactor: ...` 重构
- `test: ...` 测试
- `chore: ...` 杂项

示例：

```text
feat(ingestion): support url list in raw_pdfs
fix(agent): handle missing section filename in index.json
docs: update README for deployment
```

## 版本历史与变更记录

当前仓库未维护正式版本号/发布记录。建议在每次发布时更新以下表格（示例格式）：

| 版本 | 日期 | 变更摘要 |
| --- | --- | --- |
| v0.1.0 | 2026-02-26 | 初始可用版本：摄取 + Chainlit UI + 基础工具 |

## 许可证声明

当前仓库未包含 `LICENSE` 文件。默认情况下，除非作者另行声明，否则不授予任何复制、分发或再许可权利。若计划开源发布，请补充 `LICENSE` 并在本节更新说明。

## 常见故障排查

| 现象 | 可能原因 | 解决方案 |
| --- | --- | --- |
| 启动时报 `NVIDIA_API_KEY 未设置` | 未配置 `.env` 或环境变量 | 确认 `.env` 存在且包含 `NVIDIA_API_KEY=...`；或用 `$env:NVIDIA_API_KEY="..."` 设置 |
| `python -m chainlit ...` 报找不到模块 | 依赖未安装到当前解释器 | 确认使用 `.\.venv\Scripts\python.exe` 执行并已 `pip install -r requirements.txt` |
| 解析 URL 报 `Cannot parse PDF URL without MinerU` | URL 解析必须走 MinerU API | 配置 `MINERU_API_TOKEN`，或改用下载后的本地 PDF |
| `magic-pdf` 安装/导入失败 | 系统依赖/平台兼容问题 | 使用 PyMuPDF 回退（仅本地 PDF）；或改用 MinerU API（配置 token） |
| 摄取后 `data/processed/` 为空 | `data/raw_pdfs/` 下无 PDF/URL 列表或文件名不匹配 | 确认 `data/raw_pdfs/` 下存在 `.pdf` 或 `.txt/.urls`，并重新运行摄取命令 |
| 侧边栏不更新 | Agent 未调用 `update_workspace` 或前缀不匹配 | 确认工具输出以 `WORKSPACE_UPDATE::` 开头（见 [`tools.py`](file:///e:/work/paper-reader/src/agent/tools.py)） |
| 响应很慢/看起来卡住 | 模型推理或工具调用耗时较长 | 查看右侧 **Agent Trace** 的时间线与耗时；需要离线分析可打开 `data/memory/traces/*.jsonl` |

## Markdown 语法检查（格式验证）

本 README 使用 `pymarkdownlnt` 做 Markdown 语法检查。命令如下（可直接复制执行）：

```powershell
cd e:\work\paper-reader
.\.venv\Scripts\python.exe -m pip install -U pymarkdownlnt
.\.venv\Scripts\python.exe -m pymarkdown scan README.md
```
