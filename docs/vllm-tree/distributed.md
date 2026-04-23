# Distributed 深层推导

> 从"单卡不够"往下推，推到不能再分

---

## 表层问题

**输入**：超大模型
**输出**：多卡计算结果
**问题**：单卡显存不够怎么办？

---

## 第一层推导

```
    [单卡 → 多卡]
          │
    ┌─────┴─────┐
    │           │
[数据并行] [模型并行]
    │           │
    ▼           ▼
[DP] [TP/PP/EP]
    │           │
    └─────┬─────┘
          │
    [Distributed]
```

| 层级 | 问题 | 解决 | 概念 |
|------|------|------|------|
| 表层 | 单卡不够？ | 多卡 | Distributed |
| 底层 | 多卡如何分工 | TP/PP/EP | 并行策略 |
| 底层 | 参数如何分 | 层切分 | Sharding |
| 底层 | 通信如何优化 | NCCL | Communication |

---

## 第二层推导：并行策略

### 2.1 Tensor Parallel (TP)

```
[Tensor Parallel]
    │
    ├── 问题：单层太大？
    │       解决：层内切分
    │
    ├── 切分方式：
    │       ├── 列切分 (Column)
    │       └── 行切分 (Row)
    │
    └── 通信：AllReduce
```

### 2.2 Pipeline Parallel (PP)

```
[Pipeline Parallel]
    │
    ├── 问题：多层太多？
    │       解决：层间切分
    │
    ├── 方式：
    │       ├── 流水线
    │       └── 微批处理
    │
    └── 问题：气泡
            减少气泡
```

### 2.3 Expert Parallel (EP)

```
[Expert Parallel]
    │
    ├── MoE 专用
    │       │
    │       问题：专家太多
    │               分布到多卡
    │
    └── 通信：
            AllToAll
```

### 2.4 Data Parallel (DP)

```
[Data Parallel]
    │
    ├── 问题：想要更大batch？
    │       解决：数据切分
    │
    ├── 方式：
    │       ├── 数据复制
    │       └── gradient reduce
    │
    └── 问题：一致性
            参数同步
```

---

## 第三层推导：Sharding & Communication

### 3.1 Sharding

```
[Sharding]
    │
    ├── 问题：如何分参数？
    │       解决：1D/2D切分
    │
    ├── 问题：参数破碎？
    │       解决：分片策略
    │
    └── 问题：如何复现？
            ZeRO优化
```

### 3.2 Communication

```
[Communication]
    │
    ├── AllReduce
    │       └── TP/DP
    │
    ├── AllGather
    │       └── 收集
    │
    ├── ReduceScatter
    │       └── 聚合
    │
    └── AllToAll
            └── EP MoE
```

---

## 完整推导树

```
Distributed
    │
    ├── [Tensor Parallel]
    │       ├── 列切分
    │       ├── 行切分
    │       └── AllReduce
    │
    ├── [Pipeline Parallel]
    │       ├── 阶段切分
    │       ├── 微批处理
    │       └── 气泡最小化
    │
    ├── [Expert Parallel]
    │       ├── 专家分布
    │       └── AllToAll
    │
    ├── [Data Parallel]
    │       ├── 数据复制
    │       └── 梯度同步
    │
    └── [ZeRO]
            ├── 分片优化
            └── 通信优化
```

---

## 记住

```
Distributed = TP(层内) + PP(层间) + EP(专家) + DP(数据)
            │
            └── 核心问题：如何在多卡间合理分配 workload？
```