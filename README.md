# 基于 GAKG 知识图谱的 Acemap 搜索增强

本项目利用 GAKG（地学学术知识图谱）对 Acemap 的搜索功能进行增强。
![demo](https://cdn.jsdelivr.net/gh/ar0nkk/imgs@main/demo/image.png)

## 项目结构

```text
.
├── data/
│   └── gakg_*.parquet       # GAKG 知识图谱
├── app.py                   # 基于 Streamlit 的 Web 界面
├── main.py                  # 搜索增强主程序
├── download_data.py         # 数据下载脚本 (Hugging Face)
├── pagerank.py              # PageRank 算法实现
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
# AI_INTENT_MODEL=gpt-4o-mini   # 意图识别专用模型，优先级高于 MODEL_NAME
# AI_RAG_MODEL=gpt-4o-mini      # RAG 回答专用模型，优先级高于 MODEL_NAME
```

若使用 DashScope 兼容模式（如 DeepSeek）：
```
OPENAI_API_KEY=你的密钥
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=deepseek-v3.2-exp
# 或分别指定
# AI_INTENT_MODEL=deepseek-v3.2-exp
# AI_RAG_MODEL=deepseek-v3.2-exp
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

## 搜索增强方法

### 1. 搜索召回：查询扩展
- **目标**：先扩展关键词，从知识图谱中发现与用户术语紧密相连的相关概念，向 Acemap 发起多轮查询，召回更多相关文献。
- **步骤**：
    1. 利用 PageRank 在 GAKG 中对关键词邻域排序，选出与用户概念最紧密的 Top-3 相关词。
    2. 除了原始关键词外，还分别调用 Acemap 接口搜索这些相关概念，结果去重后合并。
- Score 计算方法：
    - 在 main.py 里，先用 PageRank 计算查询词邻域中相关概念的权重（归一化到 0~1）。
    - 对每篇论文，提取其 keywords/concepts，与这些邻域词做交集；将交集词的权重求和，得到 enhancement_score main.py。

### 2. 提高精度：图谱过滤 + 引文排序
- **目标**：在增强召回基础上，保留与知识图谱相关联且引用量高的论文。
- **步骤**：
    1. 统计每篇论文关键词/概念与拓展词集合的交叉；如果有交集，则说明和图谱邻域有关。
    2. 以引用次数为主要排序指标（`cited_by_count`），其中与图谱交叉的文献排在前面，未命中的文献在其后。

> 温馨提示：GAKG 知识图谱主要涵盖地学领域。搜索非地学关键词可能无法获得有效的增强效果（得分为 0）。
