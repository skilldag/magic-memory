# 25 - 二胡 → Forward Pass

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 25 | 🎻 二胡 | Forward Pass |

**记忆口诀**: 前向传播如二胡弦动

---

## vLLM 概念

**Forward Pass** 是前向传播：

```rust
impl ModelRunner {
    pub fn forward(&mut self, input_ids: &[Token]) -> Logits {
        // 1. Embedding
        let hidden = self.embedding.forward(input_ids);
        
        // 2. Transformer Layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &self.kv_cache);
        }
        
        // 3. Output
        self.lm_head.forward(&hidden)
    }
}
```

**记忆故事**:
- 二胡弦动 = 层层前向

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Transformer Layers](./15-parrot.md) | 15 | 鹦鹉 |
| [PagedAttention](./16-pomegranate.md) | 16 | 石榴 |

---

*二胡悠扬，前向流畅。*