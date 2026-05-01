# 5 - 钩子 → GpuAllocator

> **[← 返回目录](../level-1/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 5 | 🪝 钩子 | GpuAllocator |

**记忆口诀**: 钩子钩住 GPU 显存分配

---

## vLLM 概念

**GpuAllocator** 是 GPU 显存分配器：

```rust
pub struct GpuAllocator {
    pool: MemoryPool,
    free_blocks: Vec<CacheBlock>,
}

impl GpuAllocator {
    /// 分配块
    pub fn allocate(&mut self, num_blocks: usize) -> Vec<CacheBlock>;

    /// 释放块
    pub fn free(&mut self, blocks: Vec<CacheBlock>);
}
```

**记忆故事**:
- 钩子抓取 = 分配器抓取显存

---

## 深入理解

### 分配策略

| 策略 | 说明 | 记忆 |
|------|------|------|
| **按需分配** | 运行时分配 | 动态 |
| **预分配** | 启动时分配 | 静态 |
| **池化** | 块池复用 | 高效 |

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [CacheBlock](./18-money.md) | 18 | 钞票 |
| [KVCacheManager](./19-medicine.md) | 19 | 药酒 |

---

*钩子虽小，能挂万物。分配器虽简，能管全部显存。*