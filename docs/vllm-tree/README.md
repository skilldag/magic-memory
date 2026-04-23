# vLLM 概念树形推导

> 完整推导请看子文档

---

## 快速入口

| 主题 | 子文档 | 核心公式 |
|------|-------|----------|
| 分词器 | [tokenizer.md](./tokenizer.md) | BPE → WordPiece → SentencePiece |
| 词嵌入 | [embedding.md](./embedding.md) | 查表 + 位置编码 |
| Transformer | [transformer.md](./transformer.md) | Multi-Head + FFN + Residual + LN |
| FlashAttention | [flash-attention.md](./flash-attention.md) | Tiling + Online + Recompute |
| PagedAttention | [paged-attention.md](./paged-attention.md) | Block + Table + Ref |
| 采样器 | [sampler.md](./sampler.md) | Softmax + Temp + Top-K/P |
| 量化 | [quantization.md](./quantization.md) | Scale + 方案 + 校准 |
| 推测解码 | [speculative-decoding.md](./speculative-decoding.md) | Proposer + Verifier + Acceptance |
| 前缀缓存 | [prefix-caching.md](./prefix-caching.md) | Hash + Storage + Eviction |
| 调度器 | [scheduler.md](./scheduler.md) | Policy + Preemption + Affinity |
| 批处理 | [batching.md](./batching.md) | Static/Dynamic + Policy |
| 分布式 | [distributed.md](./distributed.md) | TP/PP/EP + Sharding + NCCL |

---

## 树形结构总览

```
根：核心循环
  │
  ├── [编码]
  │     └── Tokenizer → 详细：tokenizer.md
  │
  ├── [计算]
  │     ├── Transformer → 详细：transformer.md
  │     ├── FlashAttention → 详细：flash-attention.md
  │     └── PagedAttention → 详细：paged-attention.md
  │
  ├── [采样]
  │     └── Sampler → 详细：sampler.md
  │
  ├── [调度]
  │     ├── Scheduler → 详细：scheduler.md
  │     └── Batching → 详细：batching.md
  │
  ├── [优化]
  │     ├── Quantization → 详细：quantization.md
  │     ├── Speculative → 详细：speculative-decoding.md
  │     └── Prefix Cache → 详细：prefix-caching.md
  │
  └── [扩展]
        └── Distributed → 详细：distributed.md
```

---

## 推导层级说明

| 层级 | 描述 | 例子 |
|------|------|------|
| **表层** | 用户看到的概念 | FlashAttention |
| **第一层** | 问题的直接解决 | Tiling + Online + Recompute |
| **第二层** | 具体技术 | 分块计算、在线softmax |
| **第三层** | 实现细节 | 块大小、数值稳定 |

---

## 记住方法

```
1. 先记住根（核心循环）
2. 记住第一层分叉（编码/计算/采样/调度/优化）
3. 记住每个分叉的概念
4. 需要时再看子文档深入
```

---

*详细推导见各子文档*