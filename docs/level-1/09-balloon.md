# 9 - 气球 → KV Cache

> **[← 返回目录](../level-1/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 9 | 🎈 气球 | KV Cache |

**记忆口诀**: 气球膨胀如缓存增长

---

## vLLM 概念

**KV Cache** 是键值缓存：

```rust
pub struct KVCache {
    pub key_cache: Tensor,   // K 缓存
    pub value_cache: Tensor, // V 缓存
    pub block_table: BlockTable, // 块表映射
}
```

**记忆故事**:
- 气球膨胀 = 缓存随序列增长
- 9 = 最大 token 数相关

---

## 深入理解

### KV Cache 作用

| 阶段 | 说明 | 记忆 |
|------|------|------|
| **Prefill** | 预填充，计算并缓存全部 KV | 一次性 |
| **Decode** | 解码，从缓存读取 KV | 增量式 |

### PagedAttention

```
KV Cache
    ├── 逻辑块 (Logical Block)
    ├── 物理块 (Physical Block)
    └── 块表 (Block Table) → 映射
```

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [PagedAttention](../level-2/16-pomegranate.md) | 16 | 石榴 |
| [Block Table](../level-2/17-microscope.md) | 17 | 显微镜 |
| [CacheBlock](../level-2/18-money.md) | 18 | 钞票 |
| [KVCacheManager](../level-2/19-medicine.md) | 19 | 药酒 |

---

*气球膨胀有极限，KV Cache 膨胀有显存极限。*