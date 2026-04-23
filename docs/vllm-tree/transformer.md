# Transformer 深层推导

> 从"向量序列如何变向量序列"往下推，推到不能再分

---

## 表层问题

**输入**：向量序列 [x1, x2, ..., xn]
**输出**：向量序列 [y1, y2, ..., yn]
**问题**：如何做深层非线性变换？

---

## 第一层推导

```
    [向量序列 → 向量序列]
          │
    ┌─────┴─────┐
    │           │
[多头注意] [前馈网络]
    │           │
    ▼           ▼
[Multi-Head]  [FFN]
    │           │
    └─────┬─────┘
          │
    [Transformer]
```

| 层级 | 问题 | 解决 | 概念 |
|------|------|------|------|
| 表层 | 多层怎么叠加？ | 残差连接 | Residual |
| 底层 | 层间变化 | 归一化 | LayerNorm |
| 底层 | 注意力 | 多头 | Multi-Head Attention |
| 底层 | 非线性变换 | 两层FC | FFN |

---

## 第二层推导：Multi-Head Attention

### 2.1 Q, K, V

```
[QKV]
    │
    ├── Q (Query) - 我要找什么
    │       │
    ├── K (Key) - 我有什么
    │       │
    └── V (Value) - 我给你什么
            │
    计算：Attention(Q, K, V)
```

### 2.2 多头

```
[Multi-Head]
    │
    ├── 问题：单一注意力不够？
    │       解决：多头
    │
    ├── 做法：
    │       ├── 分割成 h 个头
    │       ├── 各自计算
    │       └── 拼接
    │
    └── 头数：32 (LLaMA)
```

### 2.3 注意力计算

```
[Attention(Q, K, V)]
    │
    ├── 1. QKᵀ → 相似度矩阵
    │       │
    ├── 2. Scale → 除以 √dk
    │       │
    ├── 3. Mask → 掩盖未来
    │       │
    └── 4. Softmax → 归一化
            │
    公式：Attention = Softmax(QKᵀ/√dk)V
```

---

## 第三层推导：FFN & Norm

### 3.1 FFN (前馈网络)

```
[FFN]
    │
    ├── 两层全连接
    │       │
    │       维度：d_model → d_ffn → d_model
    │
    ├── 激活函数
    │       ├── ReLU (原始)
    │       ├── SwiGLU (LLaMA)
    │       └── GELU (GPT)
    │
    └── 核心：提供非线性
```

### 3.2 LayerNorm

```
[LayerNorm]
    │
    ├── 问题：层间分布变化？
    │       解决：归一化
    │
    ├── 公式：y = (x - μ) / σ × γ + β
    │       │
    ├── μ = mean(x)
    │       σ = std(x)
    │       γ, β 可学习
    │
    └── 问题：训练稳定
            └── 预归一化 (Pre-LN)
```

### 3.3 Residual

```
[Residual]
    │
    ├── 问题：深层梯度消失？
    │       解决：残差连接
    │
    ├── 公式：y = F(x) + x
    │       │
    └── 核心：每层直连
```

---

## 完整推导树

```
Transformer
    │
    ├── [Multi-Head Attention]
    │       ├── QKV 投影
    │       ├── 多头分割
    │       ├── 相似度计算
    │       ├── Scale
    │       ├── Mask
    │       └── Softmax
    │
    ├── [FFN]
    │       ├── 升维
    │       ├── 激活
    │       └── 降维
    │
    ├── [Residual]
    │       └── 直连
    │
    └── [LayerNorm]
            ├── 均值归一化
            ├── 方差归一化
            └── 仿射变换
```

---

## 记住

```
Transformer = MultiHead(QKV + mask) + FFN + Residual + LayerNorm
            │
            └─��� 核心问题：如何让信息在多层中流动并变换？
```

---

## 变体速查

| 变体 | 特点 | 改动 |
|------|------|------|
| Pre-LN | 预归一化 | LayerNorm在残差前 |
| Post-LN | 后归一化 | LayerNorm在残差后 |
| SwiGLU | 门控激活 | SwiGLU激活函数 |
| RoPE | 旋转位置 | 位置编码融合 |
| FlashAttn | 高效注意力 | 替换Attention |