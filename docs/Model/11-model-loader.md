# 11 - 筷子 → ModelLoader

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 11 | 🥢 筷子 | ModelLoader |

**记忆口诀**: 筷子夹取权重，11 = 夹取

---

## vLLM 概念

**ModelLoader** 是模型加载器：

```rust
pub struct ModelLoader {
    registry: ModelRegistry,
}

impl ModelLoader {
    /// 加载 HuggingFace 模型
    pub fn load_hf_model(&self, path: &str) -> Result<ModelWeights>;

    /// 加载safetensors
    pub fn load_safetensors(&self, files: &[PathBuf]) -> Result<ModelWeights>;
}
```

**记忆故事**:
- 筷子夹起 = 加载权重

---

## 加载流程

```
ModelLoader
    ├── load_hf_model()     → HuggingFace 格式
    ├── load_safetensors()  → SafeTensors 格式
    └── load_weights()     → 权重张量
```

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [ModelRegistry](./10-baseball.md) | 10 | 棒球 |
| [Model](./12-highchair.md) | 12 | 婴儿椅 |
| [Weights Loading](./29-uncle.md) | 29 | 二舅 |

---

*筷子夹菜，也夹权重。*