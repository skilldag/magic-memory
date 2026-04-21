# 18 - 钞票 → CacheBlock

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 18 | 💵 钞票 | CacheBlock |

**记忆口诀**: CacheBlock 如钞票管理显存资源

---

## vLLM 概念

**CacheBlock** 是 KV 缓存块：

```rust
pub struct CacheBlock {
    pub block_id: i32,
    pub size: usize,  // 块大小
    pub is_free: bool,
}
```

**记忆故事**:
- 钞票管理 = 显存管理

---

## 块配置

| 参数 | 默认值 | 记忆 |
|------|--------|------|
| **block_size** | 16/32/64 | Token/块 |
| **num_blocks** | 显存/块大小 | 块数量 |

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [PagedAttention](./16-pomegranate.md) | 16 | 石榴 |
| [Block Table](./17-microscope.md) | 17 | 显微镜 |
| [KVCacheManager](./19-medicine.md) | 19 | 药酒 |

---

*钞票珍贵，显存珍贵。*