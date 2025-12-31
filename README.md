# 基于 GAKG 知识图谱的 Acemap 搜索增强

本项目利用 GAKG（地学学术知识图谱）对 Acemap 的搜索功能进行增强。
![demo](https://cdn.jsdelivr.net/gh/ar0nkk/imgs@main/demo/image.png)

## 项目结构

```text
.
├── data/
│   └── gakg_subset.parquet  # GAKG 数据子集
├── app.py                   # 基于 Flask 的 Web 界面
├── main.py                  # 搜索增强主程序
├── pagerank.py              # PageRank 算法实现
└── requirements.txt
```


## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启用搜索
- 终端运行：`main.py`
- web 端运行：`app.py`

## 搜索增强方法

### 1. 搜索召回：查询扩展
- **目标**：先扩展关键词，从知识图谱中发现与用户术语紧密相连的相关概念，向 Acemap 发起多轮查询，召回更多相关文献。
- **步骤**：
    1. 利用 PageRank 在 GAKG 中对关键词邻域排序，选出与用户概念最紧密的 Top-3 相关词。
    2. 除了原始关键词外，还分别调用 Acemap 接口搜索这些相关概念，结果去重后合并。

### 2. 提高精度：图谱过滤 + 引文排序
- **目标**：在增强召回基础上，保留与知识图谱相关联且引用量高的论文。
- **步骤**：
    1. 统计每篇论文关键词/概念与拓展词集合的交叉；如果有交集，则说明和图谱邻域有关。
    2. 以引用次数为主要排序指标（`cited_by_count`），其中与图谱交叉的文献排在前面，未命中的文献在其后。

## 注意事项

- 提供的 GAKG 子集主要涵盖地学领域。搜索非地学关键词可能无法获得有效的增强效果（得分为 0）。
- PageRank 算法复用了 `hw3-link-analysis-release` 中实现的代码。
