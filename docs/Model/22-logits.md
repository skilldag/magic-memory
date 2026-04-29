# 22 - 双胞胎 → Logits

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 22 | 👯 双胞胎 | Logits |

**记忆口诀**: Logits 如双胞胎成对出现

---

## vLLM 概念

**Logits** 是未归一化的分数：

```rust
pub struct Logits {
    pub data: Vec<f32>,  // [vocab_size]
}
```

**记忆故事**:
- 双胞胎 = vocab_size 个分数成对

---

## 处理流程

```
Hidden State → Linear → Logits → Softmax → Probabilities
```

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Sampler](./20-cigarette.md) | 20 | 香烟 |

---

*双胞胎成双成对，Logits 成 vocab 出现。*