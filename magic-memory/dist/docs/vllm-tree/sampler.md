# Sampler 深层推导

> 从"Logits如何变成Token"往下推，推到不能再分

---

## 表层问题

**输入**：Logits [3.2, 1.5, 0.8, ...]
**输出**：Token ID
**问题**：如何从分数选出一个token？

---

## 第一层推导

```
    [Logits → Token]
          │
    ┌─────┴─────┐
    │           │
[概率计算] [采样策略]
    │           │
    ▼           ▼
[Softmax]   [Top-K/P]
    │           │
    └─────┬─────┘
          │
    [Sampler]
```

| 层级 | 问题 | 解决 | 概念 |
|------|------|------|------|
| 表层 | 选哪个token？ | 采样 | Sampler |
| 底层 | 分数变概率 | Softmax | Distribution |
| 底层 | 候选太多 | Top-K | Pruning |
| 底层 | 分布截断 | Top-P | Truncation |
| 底层 | 随机性控制 | Temperature | Temperature |

---

## 第二层推导：Softmax

### 2.1 标准 Softmax

```
[Softmax]
    │
    ├── 问题：分数差太大？
    │       解决：指数放大差距
    │
    ├── 问题：指数溢出？
    │       解决：减去max (稳定softmax)
    │
    └── 公式：
            p_i = exp(logit_i) / Σ exp(logit_j)
```

### 2.2 Temperature

```
[Temperature]
    │
    ├── T > 1: 更随机
    │       ├── 问题：分布更平滑
    │       └── 解决：降低峰值
    │
    ├── T = 1: 标准
    │
    └── T < 1: 更确定
            └── 问题：分布更尖锐
                └── 解决：放大峰值
```

---

## 第三层推导：采样策略

### 3.1 Greedy

```
[Greedy]
    │
    └── 总是选最大
        │
        问题：没有随机性
        │
        解决：用于推理，后期
```

### 3.2 Top-K

```
[Top-K]
    │
    ├── 保留前K个
    │       │
    │       问题：K合适吗？
    │               动态K
    │
    └── 重新归一化
            └── 概率分布
```

### 3.3 Top-P (Nucleus)

```
[Top-P]
    │
    ├── 累积概率 ≥ P
    │       │
    │       问题：集合太小？
    │               增加P
    │
    └── 动态集合大小
            └── 取决于分布形状
```

### 3.4 Contrastive (对比采样)

```
[Contrastive]
    │
    ├── 考虑已选token
    │       │
    │       问题：避免重复
    │               惩罚已选token
    │
    └── 公式：
            logit_new = logit - α × logit_prev
```

---

## 完整推导树

```
Sampler
    │
    ├── [概率计算]
    │       ├── Softmax
    │       ├── 稳定处理
    │       └── Temperature
    │
    ├── [采样策略]
    │       ├── Greedy
    │       ├── Top-K
    │       ├── Top-P
    │       └── Contrastive
    │
    └── [后处理]
            ├── 重复惩罚
            ├── 频率惩罚
            └── 长度归一化
```

---

## 记住

```
Sampler = Softmax(温度) + Top-K/P 裁剪 + 惩罚
          │
          └── 核心问题：如何在"确定性和多样性"之间平衡？
```