# 14 - 玫瑰 → Embedding

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 14 | 🌹 玫瑰 | Embedding |

**记忆口诀**: 玫瑰芳香嵌入向量

---

## vLLM 概念

**Embedding** 是词嵌入层：

```rust
pub struct Embedding {
    num_embeddings: usize,  // 词表大小
    embedding_dim: usize,   // 嵌入维度
}

impl Embedding {
    /// 嵌入查找
    pub fn forward(&self, ids: &[TokenId]) -> Tensor;
}
```

**记忆故事**:
- 玫瑰香 = 词变向量 = 芳香

---

## 核心概念

| 概念 | 说明 | 记忆 |
|------|------|------|
| **TokenId** | 词表 ID | 词的身份证 |
| **Embedding Vector** | 嵌入向量 | 词的特征 |
| **词表大小** | vocab_size | 词有多少 |

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Model](./12-highchair.md) | 12 | 婴儿椅 |
| [Transformer Layers](./15-parrot.md) | 15 | 鹦鹉 |

---

*玫瑰芳香扑鼻，Embedding 词向量扑来。*