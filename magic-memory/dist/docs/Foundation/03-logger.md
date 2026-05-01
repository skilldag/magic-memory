# 3 - 耳朵 → Logger/Tracing

> **[← 返回目录](../level-1/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 3 | 👂 耳朵 | Logger/Tracing |

**记忆口诀**: 日志被耳朵监听，记录系统状态

---

## vLLM 概念

**Logger/Tracing** 是 vLLM 的日志追踪系统：

```rust
pub fn init_logger(config: &VllmConfig) {
    // 初始化日志系统
    tracing::info!("vLLM starting...");
    tracing::debug!("Config: {:?}", config);
}
```

**记忆故事**:
- 耳朵监听 = 日志监听
- 3 = 三种级别

---

## 日志级别

| 级别 | 用途 | 记忆 |
|------|------|------|
| **ERROR** | 错误 | 红色 |
| **WARN** | 警告 | 黄色 |
| **INFO** | 信息 | 绿色 |
| **DEBUG** | 调试 | 蓝色 |

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [VllmConfig](./00-egg.md) | 0 | 鸡蛋 |
| [Error Handling](./06-spoon.md) | 6 | 勺子 |

---

*耳朵倾听万物，日志记录一切。*