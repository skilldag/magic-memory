# 19 - 药酒 → KVCacheManager

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 19 | 🧪 药酒 | KVCacheManager |

**记忆口诀**: KVCacheManager 如药剂师分配缓存

---

## vLLM 概念

**KVCacheManager** 是 KV 缓存管理器：

```rust
pub struct KVCacheManager {
    blocks: Vec<CacheBlock>,
    allocator: GpuAllocator,
}

impl KVCacheManager {
    /// 分配块
    pub fn allocate(&mut self, num_blocks: usize) -> Vec<CacheBlock>;
    /// 释放块
    pub fn free(&mut self, blocks: Vec<CacheBlock>);
    /// 更新块表
    pub fn update_block_table(&self, seq_id: u64, table: BlockTable);
}
```

**记忆故事**:
- 药剂师配药 = Manager 分配缓存

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [CacheBlock](./18-money.md) | 18 | 钞票 |
| [KV Cache](../level-1/09-balloon.md) | 9 | 气球 |

---

*药剂师配药，Manager 配缓存。*