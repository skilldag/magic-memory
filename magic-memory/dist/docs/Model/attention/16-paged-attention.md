# 16 - 石榴 → PagedAttention

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 16 | 🫚 石榴 | PagedAttention |

**记忆口诀**: 石榴多籽 = 多分页，KV 分页管理

---

## vLLM 概念

**PagedAttention** 是 vLLM 的核心创新：

```rust
pub struct PagedAttention {
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
}

impl PagedAttention {
    pub fn forward(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_table: &Tensor,
    ) -> Tensor;
}
```

**记忆故事**:
- 石榴籽 = 分页块
- 每籽一粒 = 每块存 KV

---

## 核心思想

### 传统 Attention
```
连续内存分配
[ K ][ K ][ K ][ K ][ K ]  ← 必须连续
```

### PagedAttention
```
分页内存管理
[Block 0][Block 1][Block 2]  ← 可以非连续
```

### 优势

| 优势 | 说明 | 记忆 |
|------|------|------|
| **显存共享** | 块可复用 | 节省 |
| **灵活扩展** | 动态分配 | 弹性 |
| **前缀缓存** | Block 级复用 | 高效 |

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Block Table](./17-microscope.md) | 17 | 显微镜 |
| [CacheBlock](./18-money.md) | 18 | 钞票 |
| [KV Cache](../level-1/09-balloon.md) | 9 | 气球 |

---

*石榴籽多，KV 块多，显存利用高。*