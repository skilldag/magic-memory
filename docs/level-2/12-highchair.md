# 12 - 婴儿椅 → Model

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 12 | 🪑 婴儿椅 | Model |

**记忆口诀**: 模型本身，如婴儿被加载到 Model Runner 上

---

## vLLM 概念

**Model** 是模型抽象：

```rust
pub trait Model {
    /// 前向传播
    fn forward(&self, input: &Tensor, kv_cache: &KVCache) -> Tensor;

    /// 获取模型权重
    fn weights(&self) -> &ModelWeights;
}
```

**记忆故事**:
- 婴儿坐 = 模型加载运行

---

## 模型类型

| 模型 | 说明 | 记忆 |
|------|------|------|
| **Llama** | LLaMA 架构 | 羊驼 |
| **Mistral** | Mistral 架构 | 迷雾 |
| **Qwen** | Qwen 架构 | 青蛙 |
| **Phi** | Phi 架构 | 小而精 |

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [ModelRunner](./13-umbrella.md) | 13 | 雨伞 |
| [Transformer Layers](./15-parrot.md) | 15 | 鹦鹉 |

---

*婴儿椅上坐模型，模型上面跑推理。*