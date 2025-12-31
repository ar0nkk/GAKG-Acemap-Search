# 基于 GAKG 知识图谱的 Acemap 搜索增强

本项目利用 GAKG（地学学术知识图谱）对 Acemap 的搜索功能进行增强。
![demo](https://cdn.jsdelivr.net/gh/ar0nkk/imgs@main/demo/image.png)

## 项目结构

```text
.
├── data/
│   └── gakg_*.parquet       # GAKG 知识图谱
├── app.py                   # 基于 Flask 的 Web 界面
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

> 温馨提示：GAKG 知识图谱主要涵盖地学领域。搜索非地学关键词可能无法获得有效的增强效果（得分为 0）。
