# 20 - 香烟 → Sampler

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 20 | 🚬 香烟 | Sampler |

**记忆口诀**: Sampler 如点燃决定下一个 token

---

## vLLM 概念

**Sampler** 是采样器：

```rust
pub struct Sampler {
    temperature: f32,
    top_k: i32,
    top_p: f32,
}

impl Sampler {
    pub fn sample(&self, logits: &Logits) -> TokenId;
}
```

**记忆故事**:
- 点燃决定 = 采样决定输出

---

## 采样策略

| 策略 | 说明 | 记忆 |
|------|------|------|
| **Greedy** | 选最大 | 确定性 |
| **Temperature** | 调整分布 | 随机性 |
| **Top-K** | 选前 K | 截断 |
| **Top-P** | 累积概率 | 核采样 |

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Logits](./22-twins.md) | 22 | 双胞胎 |
| [Sampling Params](./21-crocodile.md) | 21 | 鳄鱼 |

---

*点燃一支烟，决定下个词。*