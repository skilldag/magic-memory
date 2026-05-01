# 7 - 拐杖 → Init

> **[← 返回目录](../level-1/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 7 | 🩼 拐杖 | Init |

**记忆口诀**: 初始化需要支撑，像拐杖支撑整个系统

---

## vLLM 概念

**Init** 是 vLLM 的初始化：

```rust
pub fn init_vllm(config: &VllmConfig) {
    // 初始化日志
    init_logger();
    
    // 初始化设备
    let device = Device::new_cuda(0);
    
    // 初始化显存池
    let allocator = GpuAllocator::new(gpu_memory_size);
    
    tracing::info!("vLLM initialized");
}
```

**记忆故事**:
- 拐杖支撑 = 初始化支撑全系统

---

## 初始化顺序

```
init_logger() (1)
    ↓
init_device() (2)  
    ↓
init_allocator() (3)
    ↓
init_model() (4)
    ↓
vLLM ready! ✓
```

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Logger](./03-ear.md) | 3 | 耳朵 |
| [Device](./01-candle.md) | 1 | 蜡烛 |

---

*拐杖撑起身体，init 撑起系统。*