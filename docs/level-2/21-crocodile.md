# 21 - 鳄鱼 → Sampling Params

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 21 | 🐊 鳄鱼 | Sampling Params |

**记忆口诀**: 参数如鳄鱼凶猛

---

## vLLM 概念

**Sampling Params** 是采样参数：

```rust
pub struct SamplingParams {
    temperature: f32 = 0.0,  // 温度
    top_k: i32 = 1,          // top-k
    top_p: f32 = 1.0,       // top-p
    max_tokens: usize = 1,    // 生成数
}
```

**记忆故事**:
- 鳄鱼参数凶猛 = 参数调节猛烈

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Sampler](./20-cigarette.md) | 20 | 香烟 |

---

*鳄鱼虽凶，参数调控。*