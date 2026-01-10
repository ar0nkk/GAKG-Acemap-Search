# 基于 GAKG 知识图谱的 Acemap 搜索增强

本项目利用 GAKG（地学学术知识图谱）和 LLM 对 Acemap 的搜索功能进行增强。
![demo](https://cdn.jsdelivr.net/gh/ar0nkk/imgs@main/demo/streamlit.png)

## 项目结构

```text
.
├── agent.py                 # AI 智能体
├── app.py                   # Streamlit Web 应用入口
├── config.py                # 配置文件
├── main.py                  # 主程序
├── pagerank.py              # PageRank 算法实现
├── download_data.py         # 数据下载脚本
├── run.py                   # 启动脚本
└── requirements.txt
```

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 下载知识图谱
请运行 `download_data.py` 脚本从 Hugging Face 下载所需的 Parquet 文件（约 206 MB）。

> 注：脚本会自动检测环境变量并使用您系统当前的代理配置

### 配置模型
详见 `config.py`

### 启动！
运行 `run.py` 即可

---

## Workflow

1. **User Inquiry**: 用户输入自然语言查询（如 *"板块构造理论最新研究"*）。
2. **Intent Analysis (Agent)**: LLM 分析用户意图，提取核心关键词 (`plate tectonics`) 并判断排序偏好。
3. **Knowledge Graph Expansion**: 在 GAKG 中定位关键词节点，通过 **Personalized PageRank** 算法计算相邻节点的权重，挖掘出高关联的扩展概念（如 `sediment`, `earth`）。
4. **Parallel Retrieval**: 并行向 Acemap API 发起查询（核心词 + 扩展词），获取广泛的候选论文集。
5. **Re-ranking & Scoring**: 计算每篇论文与扩展概念集的**重叠度得分 (Overlap Score)**，结合引用数和发表年份进行混合排序。
6. **AI Synthesis (RAG)**: 最后由 LLM 阅读高分论文元数据，生成结构化的研究综述和搜索建议。

## Score Calculation

论文的相关性评分 (Relevance Score) 用于衡量论文内容与查询意图的契合程度。其计算基于 GAKG 知识图谱的局部拓扑结构：

$$
\text{Score}(P) = \sum_{k \in (K_P \cap N_Q)} \text{PR}_{\text{norm}}(k)
$$

其中：
- **$P$**: 候选论文。
- **$K_P$**: 论文 $P$ 的元数据关键词集合（包含 `keywords` 和 `concepts`）。
- **$N_Q$**: 查询词 $Q$ 在 GAKG 中的**语义邻域**。该邻域通过提取 $Q$ 的一阶邻居并构建导出子图 (Induced Subgraph) 获得。
- **$\text{PR}_{\text{norm}}(k)$**: 概念 $k$ 在语义邻域子图中运行 **PageRank** 算法后得到的归一化权重（$\in [0, 1]$）。权重越高，代表该概念与查询词 $Q$ 的图谱关联越紧密。

最终，系统会优先展示能够覆盖更多、更高权重图谱概念的论文。

---

> 温馨提示：GAKG 知识图谱主要涵盖地学领域。搜索非地学关键词可能无法获得有效的增强效果（得分为 0）。