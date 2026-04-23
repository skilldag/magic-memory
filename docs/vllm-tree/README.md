# vLLM 概念树形推导

> **循环 → 问题 → 概念** 的完整推导，每个概念都有"来时路"

---

## 🎯 推荐学习路径

```
1️⃣ 先看总览图 (ASCII + Mermaid)
   └── 理解整体结构
   
2️⃣ 再看快速入口卡片
   └── 找到感兴趣的概念
   
3️⃣ 最后深入子文档
   └── 理解具体实现
```

---

## 🗂️ 文档结构

| 类型 | 文件 | 说明 |
|------|------|------|
| **主文档** | [../vLLM概念推演过程-优化版.md](../vLLM概念推演过程-优化版.md) | 完整推导+所有卡片 |
| **索引** | 本文件 | 快速入口 |
| **子文档** | 各主题子文档 | 深入理解 |

---

## 📦 快速入口

| 主题 | 子文档 | 核心公式 | 发现方法 |
|------|--------|----------|----------|
| 分词器 | [tokenizer.md](./tokenizer.md) | BPE → WordPiece → SentencePiece | 🔄 |
| 词嵌入 | [embedding.md](./embedding.md) | 查表 + 位置编码 | 🔄📉 |
| Transformer | [transformer.md](./transformer.md) | Multi-Head + FFN + Residual + LN | 🔄📉 |
| FlashAttention | [flash-attention.md](./flash-attention.md) | Tiling + Online + Recompute | 📉 |
| PagedAttention | [paged-attention.md](./paged-attention.md) | Block + Table + Ref | 📉 |
| 采样器 | [sampler.md](./sampler.md) | Softmax + Temp + Top-K/P | 🔄📉⚖️ |
| 量化 | [quantization.md](./quantization.md) | Scale + 方案 + 校准 | 📉⚖️ |
| 推测解码 | [speculative-decoding.md](./speculative-decoding.md) | Proposer + Verifier + Acceptance | 🎯 |
| 前缀缓存 | [prefix-caching.md](./prefix-caching.md) | Hash + Storage + Eviction | 🎯⚖️ |
| 调度器 | [scheduler.md](./scheduler.md) | Policy + Preemption + Affinity | ⚖️🎯 |
| 批处理 | [batching.md](./batching.md) | Static/Dynamic + Policy | ⚖️🎯 |
| 分布式 | [distributed.md](./distributed.md) | TP/PP/EP + Sharding + NCCL | 🎯📉 |
| **问题发现** | [问题发现方法论.md](./问题发现方法论.md) | 四大方法 | - |

---

## 🌳 树形结构总览

```
根：核心循环
│
├── [编码环节] 🔄
│   └── Tokenizer → Embedding
│
├── [计算环节] 🔄📉
│   ├── Transformer
│   │   │
│   │   └── Attention (核心组件)
│   │       │
│   │       ├── 📉 FlashAttention
│   │       ├── 📉 PagedAttention
│   │       └── 🎯 Streaming
│   │
│   ├── Forward Pass
│   ├── Quantization 📉⚖️
│   └── Distributed 🎯📉
│
├── [采样环节] 🔄
│   └── Sampler → Logits → Token
│
├── [调度环节] ⚖️🎯
│   ├── Scheduler
│   ├── Batching
│   └── Prefill/Decode
│
└── [优化] 📉⚖️🎯
    ├── Speculative Decoding
    └── Prefix Caching
```

---

## 🔑 发现方法图例

| 符号 | 方法 | 说明 |
|------|------|------|
| 🔄 | 流程分析 | 从工作流程自然发现 |
| 📉 | 瓶颈分析 | 从性能瓶颈倒推 |
| ⚖️ | 对比分析 | 理想vs现状差距 |
| 🎯 | 需求驱动 | 从用户需求追溯 |

---

## 📋 完整50概念索引

| # | 概念 | 层级 | 来时路 | 发现方法 |
|---|------|------|--------|----------|
| 0 | VllmConfig | 基础 | 系统配置 | 🔄 |
| 1 | Device | 基础 | 硬件抽象 | 🔄 |
| 2 | Tensor | 基础 | 数据表示 | 🔄 |
| 3 | Logger | 基础 | 日志追踪 | 🔄 |
| 4 | vllm-core | 基础 | 核心库 | 🔄 |
| 5 | GpuAllocator | 基础 | 显存分配 | 📉 |
| 6 | Error Handling | 基础 | 错误处理 | 🔄 |
| 7 | Init | 基础 | 初始化 | 🔄 |
| 8 | Foundation | 基础 | 基础层 | 🔄 |
| 9 | KV Cache | 缓存 | Attention存储 | 📉 |
| 10 | ModelRegistry | 模型 | 系统入口 | 🔄 |
| 11 | ModelLoader | 模型 | 加载权重 | 🔄 |
| 12 | Model | 模型 | 模型本体 | 🔄 |
| 13 | ModelRunner | 模型 | 执行环境 | 🔄 |
| 14 | Embedding | 编码 | Token→向量 | 🔄 |
| 15 | Transformer | 计算 | 模型核心 | 🔄 |
| 16 | PagedAttention | 计算 | Attention优化 | 📉 |
| 17 | Block Table | 计算 | 块映射 | 📉 |
| 18 | CacheBlock | 计算 | 存储单位 | 📉 |
| 19 | KVCacheManager | 计算 | 缓存管理 | 📉 |
| 20 | Sampler | 采样 | 选择token | 🔄📉 |
| 21 | Sampling Params | 采样 | 采样参数 | ⚖️ |
| 22 | Logits | 采样 | 模型输出 | 🔄 |
| 23 | Token | 采样 | 输出单元 | 🔄 |
| 24 | Decode Step | 采样 | 循环步 | 🔄 |
| 25 | Forward Pass | 计算 | 前向传播 | 🔄📉 |
| 26 | GPU Memory Pool | 基础 | 显存池 | 📉 |
| 27 | FlashAttention | 计算 | 注意力优化 | 📉 |
| 28 | Quantization | 优化 | 压缩 | 📉⚖️ |
| 29 | Weights Loading | 模型 | 加载权重 | 🔄 |
| 30 | Speculative Decoding | 优化 | 推测 | 🎯 |
| 31 | Draft Token | 优化 | 起草 | 🎯 |
| 32 | Verifier | 优化 | 验证 | 🎯 |
| 33 | N-gram Proposer | 优化 | 提议 | 🎯 |
| 34 | Batching | 调度 | 并行 | ⚖️🎯 |
| 35 | Scheduler | 调度 | 调度 | ⚖️ |
| 36 | Prefill | 调度 | 预填充 | 🎯 |
| 37 | Decode | 调度 | 解码 | 🔄 |
| 38 | Prefix Caching | 优化 | 缓存 | 🎯⚖️ |
| 39 | Request Queue | 调度 | 队列 | 🔄 |
| 40 | vllm-engine | 服务 | 引擎 | 🔄 |
| 41 | Engine API | 服务 | 接口 | 🔄 |
| 42 | vllm-serving | 服务 | 部署 | 🔄 |
| 43 | OpenAI API | 服务 | 协议 | 🔄 |
| 44 | gRPC | 服务 | 通信 | 🔄 |
| 45 | WebSocket | 服务 | 流式 | 🔄 |
| 46 | Multi-Lora | 扩展 | 多模型 | 🎯 |
| 47 | GPU Driver | 基础 | 驱动 | 🔄 |
| 48 | Prefix Lookup | 优化 | 查找 | 📉 |
| 49 | Cache Eviction | 优化 | 驱逐 | ⚖️ |
| 50 | Distributed | 扩展 | 多卡 | 🎯📉 |

---

## 🧠 记住方法

```
1. 先记住根（核心循环）
2. 记住第一层分叉（编码/计算/采样/调度）
3. 记住第二层概念（每个分叉的关键概念）
4. 需要时再看子文档深入细节

推导公式:
    来时路 → 问题 → 解决 → 概念
```

---

*详细推导见各子文档和主文档 vLLM概念推演过程-优化版.md*