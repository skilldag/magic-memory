# 0 - 鸡蛋 → VllmConfig

> **[← 返回目录](../level-1/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 0 | 🥚 鸡蛋 | VllmConfig |

**记忆口诀**: 鸡蛋是源头，所有配置的"蛋黄"

---

## vLLM 概念

**VllmConfig** 是 vLLM 的配置中心，包含：

```rust
pub struct VllmConfig {
    pub model_config: ModelConfig,      // 模型配置
    pub cache_config: CacheConfig,      // 缓存配置
    pub parallel_config: ParallelConfig, // 并行配置
    pub scheduler_config: SchedulerConfig, // 调度配置
    // ...
}
```

**记忆故事**: 
- 鸡蛋有蛋黄、蛋清、蛋壳三层
- VllmConfig 包含 model/cache/scheduler 三个配置
- 配置从「蛋」开始

---

## 深入理解

### 包含的配置

| 配置 | 说明 | 记忆 |
|------|------|------|
| **ModelConfig** | 模型路径、dtype、最大序列长度 | 模型参数 |
| **CacheConfig** | KV Cache 块大小、最大块数 | 缓存参数 |
| **ParallelConfig** | GPU 数量、PP/TP 配置 | 并行参数 |
| **SchedulerConfig** | 最大并发、队列配置 | 调度参数 |

### 配置优先级

```
VllmConfig (0-鸡蛋)
    ├── ModelConfig → 模型加载
    ├── CacheConfig → 显存分配
    ├── ParallelConfig → GPU 并行
    └── SchedulerConfig → 请求调度
```

---

## 实际使用

```rust
// 初始化配置
let config = VllmConfig::from_cli_args();

// 配置注入引擎
let engine = LlmEngine::new(config);
```

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Device 抽象](./01-candle.md) | 1 | 蜡烛 |
| [vllm-core](./04-boat.md) | 4 | 帆船 |
| [KV Cache](./09-balloon.md) | 9 | 气球 |

---

*鸡蛋虽小，却是完整生命的源头。VllmConfig 虽简单，却是整个系统的起点。*