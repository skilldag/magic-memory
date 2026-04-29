# 13 - 雨伞 → ModelRunner

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 13 | ☔ 雨伞 | ModelRunner |

**记忆口诀**: ModelRunner 如雨伞保护前向传播

---

## vLLM 概念

**ModelRunner** 是模型运行器：

```rust
pub struct ModelRunner {
    pub model: Model,
    pub kv_cache: KVCacheManager,
    pub sampler: Sampler,
}

impl ModelRunner {
    /// 执行单次前向
    pub fn forward(&mut self, input: &[Token]) -> Vec<Token>;

    /// 采样下一个 token
    pub fn sample(&mut self, logits: &Logits) -> Token;
}
```

**记忆故事**:
- 雨伞保护 = Runner 保护推理

---

## 核心组件

```
ModelRunner
    ├── model: Model        → 运行模型
    ├── kv_cache: KVCache   → 缓存管理
    └── sampler: Sampler     → 采样器
```

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Model](./12-highchair.md) | 12 | 婴儿椅 |
| [Forward Pass](./25-erhu.md) | 25 | 二胡 |
| [PagedAttention](./16-pomegranate.md) | 16 | 石榴 |

---

*雨伞遮风雨，Runner 护推理。*