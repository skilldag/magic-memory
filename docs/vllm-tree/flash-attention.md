# FlashAttention 深层推导

> 从"Attention如何变快"往下推，推到不能再分

---

## 表层问题

**输入**：Q, K, V 向量
**输出**：Attention(Q,K,V)
**问题**：Attention计算太慢(O(N²))怎么办？

---

## 第一层推导

```
    [Attention 计算问题]
           │
    ┌─────┴─────┐
    │           │
[显存O(N²)] [计算O(N²)]
    │           │
    ▼           ▼
[分块计算]  [在线计算]
    │           │
    └─────┬─────┘
          │
    [FlashAttention]
```

| 层级 | 问题 | 解决 | 概念 |
|------|------|------|------|
| 表层 | Attention 计算慢 | 用高效算法 | FlashAttention |
| 底层 | 显存O(N²) | 分块tiling | Tiling |
| 底层 | 需要存储QKᵀ | 在线计算 | Online Softmax |
| 底层 | 梯度存储 | 重新计算 | Recomputation |

---

## 第二层推导：核心技术

### 2.1 Tiling (分块)

```
[分块计算]
    │
    ├── 问题：长序列显存爆炸？
    │       解决：切成小块
    │
    ├── 问题：小块如何计算？
    │       解决：逐块计算
    │
    └── 问题：如何合并结果？
            解决：缩放求和
```

**公式**:
```
output = Σ(scale_i × block_i) / Σ(scale_i)
```

### 2.2 Online Softmax

```
[在线计算]
    │
    ├── 问题：需要存中间矩阵？
    │       解决：不存，边算边更新
    │
    ├── 问题：Softmax需要全局max？
    │       解决：在线追踪max
    │
    └── 问题：数值溢出？
            解决：减去max
```

**公式**:
```
m_i = max(m_{i-1}, row_i)
f_i = exp(row_i - m_i)
l_i = l_{i-1} + sum(f_i)
output = f_i / l_i
```

### 2.3 Recomputation (重计算)

```
[重计算]
    │
    ├── 问题：反向需要中间结果？
    │       解决：不存，前向重算
    │
    ├── 问题：重算很慢？
    │       解决：小块重算
    │
    └── 问题：哪些需要存？
            只有 m_i, l_i, output
```

---

## 第三层推导：实现细节

### 3.1 前向传播流程

```
1. Q 分割成 T 个块
2. For each block:
    a. 读取 Q_block, K, V
    b. 计算 QKᵀ (tiling)
    c. Online softmax
    d. 计算输出块
    e. 更新全局 exp sum
3. 合并所有块
```

### 3.2 反向传播流程

```
1. 需要存什么：
   - output (Y)
   - max (m)
   - exp sum (l)
2. 不需要存：
   - QKᵀ 中间矩阵
3. 重算：
   - 前向时重新计算每个块
```

---

## 完整推导树

```
FlashAttention
    │
    ├── [Tiling - 分块计算]
    │       ├── 块大小选择
    │       ├── 逐块读取
    │       └── 缩放合并
    │
    ├── [Online Softmax - 在线计算]
    │       ├── 追踪max
    │       ├── 指数求和
    │       └── 数值稳定
    │
    └── [Recomputation - 重计算]
            ├── 只存必要状态
            ├── 重算中间
            └── 显存换计算
```

---

## 记住

```
FlashAttention = Tiling + Online Softmax + Recomputation
              │
              └── 核心问题：如何用 O(N) 显存完成 O(N²) 计算？
```