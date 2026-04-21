# 29 - 二舅 → Weights Loading

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 29 | 👨 二舅 | Weights Loading |

**记忆口诀**: load_hf_model 如二舅帮忙加载

---

## vLLM 概念

**Weights Loading** 是权重加载：

```rust
pub fn load_hf_model(path: &str) -> Result<ModelWeights> {
    // 1. 加载 safetensors
    // 2. 转换为目标 dtype
    // 3. 加载到 GPU
}
```

**记忆故事**:
- 二舅帮忙 = 帮忙加载

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [ModelLoader](./11-chopsticks.md) | 11 | 筷子 |

---

*二舅帮忙，权重加载。*