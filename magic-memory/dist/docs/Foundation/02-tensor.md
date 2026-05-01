# 2 - 鸭子 → Tensor

> **[← 返回目录](../level-1/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 2 | 🦆 鸭子 | Tensor |

**记忆口诀**: Tensor 如鸭子漂浮在 GPU 内存上

---

## vLLM 概念

**Tensor** 是 vLLM 的张量抽象：

```rust
pub struct Tensor {
    pub data: *mut f16,  // 数据指针
    pub shape: [i32; 4], // 形状 [batch, seq, head, dim]
    pub device: Device,    // 所在设备
    pub layout: TensorLayout, // 内存布局
}
```

**记忆故事**:
- 鸭子浮在水面 = Tensor 漂浮在 GPU 内存
- 2 = 两个核心：data + shape

---

## 深入理解

### 张量形状

| 维度 | 含义 | 记忆 |
|------|------|------|
| **Batch** | 批次大小 | 并行请求数 |
| **Seq** | 序列长度 | 单请求 token 数 |
| **Head** | 注意力头 | 多头数 |
| **Dim** | 头维度 | 隐藏层大小 |

### 常见张量

```
Tensor
    ├── Q (Query)          → 查询向量
    ├── K (Key)          → 键向量
    ├── V (Value)        → 值向量
    └── Logit           → 输出 logits
```

---

## 实际使用

```rust
// 创建注意力 QKV
let qkv = Tensor::new(shape, Device::cuda(0));

// 批量创建
let q = split(&qkv, 0, num_heads);
let k = split(&qkv, num_heads, num_heads);
let v = split(&qkv, num_heads * 2, num_heads);
```

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Device](./01-candle.md) | 1 | 蜡烛 |
| [Embedding](./14-rose.md) | 14 | 玫瑰 |
| [PagedAttention](./16-pomegranate.md) | 16 | 石榴 |

---

*鸭子划水看似悠闲，实则水下不停扑腾。Tensor 看似简单，实则承载全部计算。*