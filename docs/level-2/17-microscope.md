# 17 - 显微镜 → Block Table

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 17 | 🔬 显微镜 | Block Table |

**记忆口诀**: Block Table 如显微镜看物理块映射

---

## vLLM 概念

**Block Table** 是块表映射：

```rust
pub struct BlockTable {
    data: Vec<i32>,  // 物理块 ID
}

impl BlockTable {
    /// 获取物理块
    pub fn get_physical_block(&self, logical_idx: usize) -> i32;

    /// 设置映射
    pub fn set_block(&mut self, logical_idx: usize, physical_idx: i32);
}
```

**记忆故事**:
- 显微镜放大 = 映射关系

---

## 映射关系

```
逻辑块 → Block Table → 物理块
[0][1][2][3]  →  [5][3][1][9]  → GPU 显存
```

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [PagedAttention](./16-pomegranate.md) | 16 | 石榴 |
| [CacheBlock](./18-money.md) | 18 | 钞票 |

---

*显微镜看映射，Block Table 找物理块。*