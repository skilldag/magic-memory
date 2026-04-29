# 6 - 勺子 → Error Handling

> **[← 返回目录](../level-1/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 6 | 🥄 勺子 | Error Handling |

**记忆口诀**: 勺子舀起错误 VllmError

---

## vLLM 概念

**Error Handling** 是 vLLM 的错误处理：

```rust
pub enum VllmError {
    NotFound(String),      // 资源未找到
    InvalidInput(String),  // 输入无效
    RuntimeError(String), // 运行时错误
    OutOfMemory(String),  // 显存不足
}

impl std::error::Error for VllmError {}
```

**记忆故事**:
- 勺子舀起 = 捕获错误

---

## 错误类型

| 错误 | 场景 | 记忆 |
|------|------|------|
| **NotFound** | 模型/权重不存在 | 找不到 |
| **InvalidInput** | 参数错误 | 输入错 |
| **RuntimeError** | CUDA 错误 | 运行错 |
| **OutOfMemory** | 显存不足 | 显存爆 |

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Logger](./03-ear.md) | 3 | 耳朵 |

---

*勺子舀起汤，也舀起错误。*