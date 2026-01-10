# 基于 GAKG 知识图谱的 Acemap 搜索增强

本项目利用 GAKG（地学学术知识图谱）和 LLM 对 Acemap 的搜索功能进行增强。
![demo](https://github.com/ar0nkk/imgs/blob/main/demo/streamlit.png)
- 数据集下载：[GAKG dataset](https://huggingface.co/datasets/aronkk/gakg)
- Acemap 原地址：[Acemap](https://acemap.info)

## 项目结构

```text
.
├── agent.py                 # AI 智能体
├── app.py                   # Web 应用入口
├── config.py                # 配置文件
├── main.py                  # 主程序
├── pagerank.py              # PPR 算法实现
├── download_data.py         # 数据下载脚本
├── process_data.py          # 数据清洗脚本
├── run.py                   # 启动脚本
└── requirements.txt
```

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 下载知识图谱
请运行 `download_data.py` 脚本从 Hugging Face 下载已清洗数据的 Parquet 文件（约 47 MB）。

> 注：脚本会自动检测环境变量并使用您系统当前的代理配置

> 如果使用其他知识图谱且需要清洗数据，请运行 `process_data.py` 并删除 `data` 中清洗前的数据

### 配置模型
详见 `config.py`

### 启动！
运行 `run.py` 即可

---

## 🚀 Workflow

1. 用户输入自然语言查询（如 *"板块运动理论最新研究"*）。
2. LLM 分析用户搜索意图，提取核心关键词（如 `plate tectonics`）
3. KG Expansion: 
   - 在 GAKG 中定位核心词，并向外扩展 1-2 跳，构建局部语境子图。
   - 采用 Context-Aware PPR 算法计算子图中各节点的结构权重。
   - 应用 Global Degree Penalty 惩罚高频通用词（如 "analysis", "study"），降低这类过于宽泛的概念的分数。
4. Hybrid Search: 并行向 API 发起查询，同时获取核心词和高权重扩展词（如 `subduction`）的论文。
5. Re-ranking & Scoring: 
   - KG Overlap Score 计算:

     $$S_{GAKG} = \sum_{w \in K_{paper} \cap V_{subgraph}} S_{PPR}(w)$$

     即：若论文的关键词 $w$ 出现在查询词的扩展子图节点集 $V_{subgraph}$ 中，则累加该词的 PageRank 权重。

   - 综合排序评分公式: 

     $$Score_{Final} = S_{GAKG} + \alpha \times \log(\text{Citations} + 1)$$

     其中 $\alpha$ 为影响力平衡系数（默认为 0.2，可在 `config.py` 中调整）。
6. Agent Synthesis (RAG): LLM 阅读高分论文的元数据，生成带引用的综述回答。

### Context-Aware PageRank 算法细节

传统的搜索扩展往往引入噪音。本项目采用改进的 Context-Aware PageRank 策略来确保扩展词的精准度：

#### 1. 局部子图个性化 (Subgraph Personalization)
不同于全局 PageRank，我们只在查询词的 **2-hop 邻域** 内构建稀疏子图。这相当于将 "Personalization Vector" 的概率质量集中在查询词及其邻居上，确保了计算出的权重完全服务于当前的局部语境，而非全局重要性。

#### 2. 全局度数惩罚机制 (Global Degree Penalty)
"Model", "System", "Area" 等无价值节点在图谱中度数极高，容易吸走 PageRank 权重。为了解决 PageRank 发散到通用词（Super Nodes）的问题，我们引入了 INF 思想：

  $$S_{final}(v) = S_{PageRank}(v) \times \log \left( \frac{N_{total}}{Degree_{global}(v) + \epsilon} \right)$$
如果一个词在全图中连接了太多其他词（说明它是通用词），它的最终得分会被大幅削减。

---

> 温馨提示：GAKG 知识图谱主要涵盖地学领域。搜索非地学关键词可能无法获得有效的增强效果（得分为 0）。