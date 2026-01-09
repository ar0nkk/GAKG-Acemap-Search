# 基于 GAKG 知识图谱的 Acemap 搜索增强

本项目利用 GAKG（地学学术知识图谱）对 Acemap 的搜索功能进行增强。
![demo](https://cdn.jsdelivr.net/gh/ar0nkk/imgs@main/demo/image.png)

## 项目结构

```text
.
├── data/
│   └── gakg_*.parquet       # GAKG 知识图谱核心数据
├── app.py                   # Streamlit Web 应用入口
├── ai_intent.py             # LLM 意图识别与 RAG 助手模块
├── config.py                # 全局配置与环境加载
├── main.py                  # 搜索、图计算与数据加载核心逻辑
├── pagerank.py              # PageRank 算法实现
├── download_data.py         # 数据下载脚本
├── run.py                   # 启动助手
└── requirements.txt
```

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 下载知识图谱
由于数据集较大，请运行 `download_data.py` 脚本从 Hugging Face 下载所需的 Parquet 文件。

> 注：脚本会自动检测并使用您系统当前的代理配置（环境变量或 Windows 系统代理）。

### 启用搜索
- 简便启动：`python run.py`（自动在项目根目录调起 Streamlit）
- 终端 CLI：`python main.py`
- Web：`streamlit run app.py`

### 新功能
- 结果展示作者与机构信息。
- 支持按作者、机构过滤，或与关键词组合检索（表单中的可选输入）。

### AI 搜索 + RAG 科研助手
- 新增基于 LLM 的意图理解：自动识别“最新/引用最多”等排序意图，并将自然语言查询（如 “recent research results on plate tectonics”）转为核心检索词。
- 新增 RAG 科研助手：结合知识图谱扩展词、检索结果，为科研场景生成简洁回答和下一步建议。

运行前请准备 `.env`（位于项目根目录）：
```
OPENAI_API_KEY=你的密钥
# 可选：自定义 API Base / 模型
# OPENAI_API_BASE=https://...
# MODEL_NAME=gpt-4o-mini        # 全局模型名称
```

若使用 DashScope 兼容模式（如 DeepSeek）：
```
OPENAI_API_KEY=你的密钥
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=deepseek-v3.2-exp
```

`MODEL_API_BASE` 也被接受为 `OPENAI_API_BASE` 的别名。

## 性能优化方法

本项目包含以下关键性能优化，以确保在本地环境下的流畅体验：

1.  **GAKG 数据轻量化加载**:
    -   在 `main.py` 的 `load_gakg` 函数中，我们仅读取 Parquet 文件中的核心列 (`subject`, `object`)，丢弃了未使用的列（如关系类型等）。这显著降低了内存占用和 I/O 开销。
    -   **预处理**: 数据加载后立即对 `subject` 和 `object` 列执行 `str.lower()` 归一化。这避免了在每次搜索查询时重复进行昂贵的全表字符串转换操作，大幅提升了图遍历的速度。

2.  **并发 API 请求**:
    -   在搜索过程中，系统需要同时对用户的主关键词以及 GAKG 扩展出的多个相关概念进行 Acemap API 查询。
    -   我们在 `app.py` 中引入了 `ThreadPoolExecutor`，并发执行这些独立的 HTTP 请求。这消除了串行网络等待时间，使得多源搜索的总耗时接近于单次最慢请求的耗时。

3.  **UI 渲染优化**:
    -   使用 Streamlit 的 `@st.cache_resource` 装饰器对庞大的 GAKG 数据进行全局缓存，确保只有在应用首次启动时需要加载数据，后续刷新或重搜均瞬间完成。
4.  **搜索与邻域缓存**:
    -   对 GAKG 邻域计算与 Acemap 查询增加 LRU 缓存，重复关键词或排序请求可命中缓存，减少 API 往返与图计算时间。

## 知识图谱清洗与去噪
- **清洗位置**：在 `load_gakg` 内置执行（见 `main.py` 的 `clean_gakg`）。
- **步骤**：
    1) 归一化大小写与空白；
    2) 去除空值与自环；
    3) 基于长度的噪声过滤（过短/过长的 token）；
    4) 去重边；
    5) 度阈值剪枝（仅保留两端度数 ≥ 2 的边，默认可调整 `min_degree`）。
- **效果**：减少孤立点与重复噪声边，压缩存储体积并提升后续查询/扩展的稳定性。

## 核心工作流与排序算法 (Core Workflow & Algorithms)

系统采用 LLM 意图识别与 GAKG 知识图谱相结合的方式，实现精确且具有上下文感知的论文检索。完整流程如下：

### 1. 意图识别 (Intent Parsing)
- **输入**：用户的自然语言查询（如 "latest papers on deep learning"）。
- **处理**：使用 LLM (`ai_intent.py`) 解析出：
    - **核心检索词** (`keyword`)：用于 API 搜索。
    - **排序意图** (`sort`)：识别用户偏向 "引用最高" (Most Cited) 还是 "最新发表" (Latest Published)。
- **效果**：将模糊的自然语言转化为精确的结构化检索指令。

### 2. 图谱扩展与召回 (GAKG Expansion & Retrieval)
- **原理**：基于 PageRank 算法在 GAKG 图谱中计算核心检索词的邻域节点权重。
- **扩展**：选取权重最高的 Top-3 相关概念（如搜索 "plate tectonics" 扩展出 "subduction", "lithosphere"）。
- **召回**：并发调用 Acemap API，分别搜索核心词和扩展词，合并结果并去重。
- **阈值控制**：用户可通过滑块调节 `neighbor_threshold`。阈值越高，过滤越严格，接近 1.0 时仅使用核心词搜索。

### 3. 相关性打分 (Scoring: Enhancement Score)
为了量化论文与搜索主题的关联深度，系统计算 **Enhancement Score**：
- **权重计算**：利用 PageRank 计算出的扩展词权重（`weighted_neighbors`），归一化至 0~1。
- **论文得分**：遍历每篇召回论文的 `keywords` 和 `concepts` 字段：
  $$ \text{Score} = \sum_{w \in (\text{PaperKeywords} \cap \text{Neighbors})} \text{Weight}(w) $$
  即：论文包含的关键词若出现在图谱扩展词中，则累加对应的 PageRank 权重。得分越高，代表该论文在知识图谱的语义空间中越核心。

### 4. 分层排序策略 (Hierarchical Sorting)
系统采用 **两级排序** 逻辑，优先展示语义更相关的结果：
1.  **第一级：图谱重叠 (Graph Overlap)**
    -   **相关结果 (Relevant)**：`enhancement_score > 0` 的论文（即命中知识图谱扩展词）。
    -   **其他结果 (Others)**：未命中图谱扩展词的论文。
    -   *界面上分为 "✨ Graph-overlap results" 和 "📄 No graph overlap" 两个区块展示。*
2.  **第二级：用户偏好 (User Preference)**
    -   在第一级的每个分组内部，根据用户选择的排序方式进行二次排序：
        -   **Most Cited**：按 `cited_by_count` 降序。
        -   **Latest Published**：按 `publication_date` 或 `publication_year` 降序。

### 5. RAG 智能总结 (RAG Assistant)
- **输入**：用户的原始问题 + Top 检索结果 + 图谱扩展词。
- **生成**：利用 LLM (`ai_intent.py`) 阅读这些上下文，生成针对该研究问题的回答、关键文献引用及后续研究建议。

> 温馨提示：GAKG 知识图谱主要涵盖地学领域。搜索非地学关键词可能无法获得有效的增强效果（得分为 0）。
