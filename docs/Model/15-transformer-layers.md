# 15 - 鹦鹉 → Transformer Layers

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 15 | 🦜 鹦鹉 | Transformer Layers |

**记忆口诀**: 鹦鹉学舌对应多层 Transformer

---

## vLLM 概念

**Transformer Layers** 是 Transformer 层：

```rust
pub struct TransformerLayer {
    pub attn: MultiHeadAttention,  // 注意力
    pub mlp: MlpLayer,              // MLP
    pub norm_q: LayerNorm,          // Q 归一化
    pub norm_k: LayerNorm,           // K 归一化
}
```

**记忆故事**:
- 鹦鹉学舌 = 层层传递

---

## 层结构

```
Transformer Layer
    ├── Norm (Pre-Norm)
    ├── Attention (MHA)
    ├── Norm
    └── MLP (SwiGLU/GELU)
```

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Embedding](./14-rose.md) | 14 | 玫瑰 |
| [PagedAttention](./16-pomegranate.md) | 16 | 石榴 |

---

*鹦鹉学舌，学的是 Transformer。*