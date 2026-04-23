# vLLM 概念推导过程（树形结构）

> 以**最小循环为根**，**问题为边**，**概念为节点**的树形推导
> 
> **问题发现方法标注**：🔄 流程分析 | 📉 瓶颈分析 | ⚖️ 对比分析 | 🎯 需求驱动

---

## 1. 根：最小循环

```
     [自回归生成循环]
            │
    ┌───────┴───────┐
    │               │
 [编码] → [计算] → [采样] → [判断] → (回到编码)
```

**这就是 vLLM 的根，没有它就无法工作。**
🔄 **发现方法**：流程分析（从工作流程自然推导出核心循环）

---

## 2. 第一层展开：每个环节的问题

```
                    [核心循环]
                    ┌─────┴─────┐
                    │           │
              ┌─────┴─────┐    │
              │           │    │
         [编码问题]   [计算问题]  ... 
              │           │
         🔄   Tokenizer    🔄  Transformer
              │              │
                     ▼
              🔄  Attention ──→ (需要解决问题)
```

| 环节 | 问题 | 发现方法 | 引入概念 |
|------|------|----------|----------|
| 编码 | 输入如何处理？ | 🔄 流程分析 | Tokenizer, Embedding |
| 计算 | 模型如何计算？ | 🔄 流程分析 | Model, Transformer |
| 采样 | 如何选next token？ | 🔄 流程分析 | Sampler, Logits |
| 调度 | 多请求怎么办？ | ⚖️ 对比分析 | Scheduler, Batching |

---

## 3. 第二层展开：一个概念需要多个问题

### 3.1 Attention 问题的分叉

```
         [Attention 计算] 📉🔄
                  │
     ┌─────────────┼─────────────┐
     │             │             │
 [计算量太大]   [显存碎片]   [长序列]
     │             │             │
 📉 瓶颈分析  📉 瓶颈分析  🎯 需求驱动
     ▼             ▼             ▼
 FlashAttention  PagedAttention  Streaming
```

**发现问题的方法**：

| 问题 | 发现方法 | 原因 |
|------|----------|------|
| 计算量太大 | 📉 瓶颈分析 | 测试发现O(N²)太慢 |
| 显存碎片化 | 📉 瓶颈分析 | 显存监控发现碎片 |
| 长序列处理 | 🎯 需求驱动 | 用户需要长上下文 |

### 3.2 显存管理问题的分叉

```
          [显存管理] 📉⚖️
               │
     ┌─────────┼─────────┐
     │         │         │
 [分配问题] [缓存问题] [复用问题]
     │         │         │
 📉 瓶颈分析 ⚖️ 对比分析 🎯 需求驱动
     ▼         ▼         ▼
 Allocator   KV Cache  Prefix Cache
```

### 3.3 调度问题的分叉

```
          [多请求调度] ⚖️🎯
               │
     ┌──────────┼──────────┐
     │          │          │
 [批处理]  [优先级]  [阶段分离]
     │          │          │
 ⚖️ 对比分析 ⚖️ 对比分析 🎯 需求驱动
     ▼          ▼          ▼
 Continuous   Priority   Prefill/
 Batching     Queue      Decode
```
     [自回归生成循环]
            │
    ┌───────┴───────┐
    │               │
 [编码] → [计算] → [采样] → [判断] → (回到编码)
```

**这就是 vLLM 的根，没有它就无法工作。**

---

## 2. 第一层展开：每个环节的问题

```
                    [核心循环]
                    ┌─────┴─────┐
                    │           │
              ┌─────┴─────┐    │
              │           │    │
         [编码问题]   [计算问题]  ... 
              │           │
         Tokenizer    Transformer
         Embedding       │
                         ▼
                   Attention ──→ (需要解决问题)
```

| 环节 | 问题 | 引入概念 |
|------|------|----------|
| 编码 | 输入如何处理？ | Tokenizer, Embedding |
| 计算 | 模型如何计算？ | Model, Transformer |
| 采样 | 如何选next token？ | Sampler, Logits |
| 调度 | 多请求怎么办？ | Scheduler, Batching |

---

## 3. 第二层展开：一个概念需要多个问题

### 3.1 Attention 问题的分叉

```
         [Attention 计算]
                  │
     ┌─────────────┼─────────────┐
     │             │             │
 [计算量太大]   [内存碎片]   [长序列]
     │             │             │
     ▼             ▼             ▼
 FlashAttention  PagedAttention  Streaming
```

**一个 Attention，引出三个问题：**

| 问题 | 解决 | 概念 |
|------|------|------|
| 计算量太大 | 减少计算复杂度 | FlashAttention |
| 显存碎片化 | 分页管理 | PagedAttention |
| 长序列处理 | 流式处理 | StreamingAttention |

### 3.2 显存管理问题的分叉

```
         [显存管理]
              │
    ┌─────────┼─────────┐
    │         │         │
[分配问题] [缓存问题] [复用问题]
    │         │         │
    ▼         ▼         ▼
Allocator   KV Cache  Prefix Cache
```

---

## 3.3 调度问题的分叉

```
          [多请求调度]
               │
    ┌──────────┼──────────┐
    │          │          │
[批处理]  [优先级]  [阶段分离]
    │          │          │
    ▼          ▼          ▼
Continuous   Priority   Prefill/
Batching     Queue      Decode
```

---

## 4. 完整问题网络

```
                    ┌─────────────────────────────────────────┐
                    │           [自回归生成循环]             │
                    │              输入→输出                 │
                    └─────────────────┬───────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
        ┌─────┴─────┐           ┌─────┴─────┐           ┌─────┴─────┐
        │           │           │           │           │           │
    [编码问题]   [计算问���]   [采样问题]   [调度问题]   [优化问题]
        │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼
    Tokenizer    Model       Sampler     Scheduler    FlashAttention
    Embedding    Transformer Logits     Batching    Quantization
                   │                            │
                   ▼                            ▼
             Attention                    Speculative
                   │                            │
        ┌──────────┼──────────┐           Prefix Cache
        │          │          │
    [计算问题] [内存问题] [长序列]
        ▼          ▼          ▼
   FlashAttn  PagedAttn  Streaming
```

---

## 5. 问题→概念映射表

### 编码阶段

| 问题 | 解决 | 概念 |
|------|------|------|
| 文本→数值 | 分词 | Tokenizer (23) |
| ID→向量 | 嵌入 | Embedding (14) |

### 计算阶段

| 问题 | 解决 | 概念 |
|------|------|------|
| 模型计算 | 前向传播 | Forward Pass (25) |
| 多层堆叠 | Transformer | Transformer (15) |
| 计算慢 | 高效算法 | FlashAttention (27) |
| 显存碎片 | 分页管理 | PagedAttention (16) |
| 物理映射 | 块表 | Block Table (17) |
| 物理块 | 存储单位 | CacheBlock (18) |

### FlashAttention 深层推导

FlashAttention 本身也是推导出来的，不是凭空出现的：

```
    [Attention 计算问题]
           │
    ┌─────┴─────┐
    │           │
[显存O(N²)] [计算O(N²)]
    │           │
    ▼           ▼
[分块计算]  [在线计算]
    │           │
    └─────┬─────┘
          │
    [FlashAttention]
```

| 层级 | 问题 | 解决 | 概念 |
|------|------|------|------|
| 表层 | Attention 计算慢 | 用高效算法 | FlashAttention |
| 底层 | 显存O(N²) | 分块tiling | Tiling |
| 底层 | 需要存储QKᵀ | 在线计算 | Online Softmax |
| 底层 | 梯度存储 | 重新计算 | Recomputation |

**FlashAttention = Tiling + Online Softmax + Recomputation**

这三个底层技术才是真正的"实现层"。

### 采样阶段

| 问题 | 解决 | 概念 |
|------|------|------|
| 分数→token | 采样 | Sampler (20) |
| 控制生成 | 参数 | Sampling Params (21) |
| 原始输出 | 分数 | Logits (22) |

### 调度阶段

| 问题 | 解决 | 概念 |
|------|------|------|
| 多请求 | 批处理 | Continuous Batching (34) |
| 请求排队 | 队列 | Request Queue (39) |
| 优先级 | 调度 | Scheduler (35) |
| 阶段区分 | 分离 | Prefill (36) + Decode (37) |

### 优化阶段

| 问题 | 解决 | 概念 |
|------|------|------|
| 自回归慢 | 推测 | Speculative Decoding (30) |
| 前缀重复 | 缓存 | Prefix Caching (38) |
| 参数太大 | 压缩 | Quantization (28) |

### PagedAttention 深层推导

```
    [KV Cache 显存问题]
           │
    ┌──────┴──────┐
    │             │
[预分配固定] [不连续]
    │             │
    ▼             ▼
[动态分页]   [块表映射]
    │             │
    └──────┬──────┘
           │
    [PagedAttention]
```

| 层级 | 问题 | 解决 | 概念 |
|------|------|------|------|
| 表层 | 显存碎片化 | 分页管理 | PagedAttention |
| 底层 | 预分配固定长度 | 动态分配 | Block Allocation |
| 底层 | 不连续内存 | 块表映射 | Block Table |
| 底层 | 物理块管理 | 引用计数 | Ref Count |

**PagedAttention = Block + Block Table + Ref Counting**

### 服务阶段

| 问题 | 解决 | 概念 |
|------|------|------|
| 如何服务 | API | Engine API (41) |
| 协议 | OpenAI | OpenAI API (43) |
| 流式 | WebSocket | WebSocket (45) |

---

## 9. 深层推导：所有概念的底层问题

每个表层概念，继续往下推，直到"不能再分"为止。

### 9.1 Tokenizer (23) - 分词器 🔄

```
    [文本→Token] 🔄 流程分析
          │
    ┌─────┴─────┐
    │           │
 [分词方式] [编码方式]
    │           │
    ▼           ▼
 [规则/MER] [ID映射]
    │           │
    └─────┬─────┘
          │
    [Tokenizer]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | 文本切成什么？ | 🔄 流程分析 | 分词 | Tokenizer |
| 底层 | 分词算法 | 📉 瓶颈分析 | BPE/WordPiece/SentencePiece | 分词模式 |
| 底层 | 词表太大 | ⚖️ 对比分析 | 频次过滤 | Vocab Pruning |
| 底层 | 未登录词 | 📉 瓶颈分析 | Byte-level | BPE |

**Tokenizer = 分词模式 + 词表 + Byte fallback**

### 9.2 Embedding (14) - 词嵌入 🔄

```
    [Token ID → 向量] 🔄 流程分析
          │
    ┌─────┴─────┐
    │           │
 [语义向量] [位置信息]
    │           │
    ▼           ▼
 [词嵌入矩阵]  [位置编码]
    │           │
    └─────┬─────┘
          │
    [Embedding]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | ID怎么变向量？ | 🔄 流程分析 | 查表 | Embedding |
| 底层 | 向量表示什么？ | 🔄 流程分析 | 语义相似 | 词向量 |
| 底层 | 位置信息丢失？ | 📉 瓶颈分析 | 加位置编码 | Positional Encoding |
| 底层 | 位置编码类型 | 📉 瓶颈分析 | 绝对/相对/RoPE | 编码类型 |

**Embedding = 词表矩阵 + 位置编码函数**

### 9.3 Transformer (15) - Transformer层 🔄📉

```
    [向量序列 → 向量序列] 🔄 流程分析
          │
    ┌─────┴─────┐
    │           │
 [多头注意] [前馈网络]
    │           │
    ▼           ▼
 [Multi-Head]  [FFN]
    │           │
    └─────┬─────┘
          │
    [Transformer]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | 多层怎么叠加？ | 🔄 流程分析 | 残差连接 | Residual |
| 底层 | 层间变化 | 📉 瓶颈分析 | 归一化 | LayerNorm |
| 底层 | 注意力 | 🔄 流程分析 | 多头 | Multi-Head Attention |
| 底层 | 非线性变换 | 📉 瓶颈分析 | 两层FC | FFN |

**Transformer = Multi-Head + FFN + Residual + LayerNorm**

### 9.4 Sampler (20) - 采样器 🔄📉⚖️

```
    [Logits → Token] 🔄 流程分析
          │
    ┌─────┴─────┐
    │           │
 [概率计算] [采样策略]
    │           │
    ▼           ▼
 [Softmax]   [Top-K/P]
    │           │
    └─────┬─────┘
          │
    [Sampler]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | 选哪个token？ | 🔄 流程分析 | 采样 | Sampler |
| 底层 | 分数变概率 | 📉 瓶颈分析 | Softmax | Distribution |
| 底层 | 候选太多 | ⚖️ 对比分析 | Top-K | Pruning |
| 底层 | 分布截断 | ⚖️ 对比分析 | Top-P | Truncation |
| 底层 | 随机性控制 | ⚖️ 对比分析 | Temperature | Temperature |

**Sampler = Softmax + Temperature + Top-K/P**

### 9.5 Forward Pass (25) - 前向传播

```
    [输入 → 输出]
│
    ┌─────┴─────┐
    │           │
 [单层计算] [混合精度]
    │           │
    ▼           ▼
 [层层堆叠] [FP16/BF16]
    │           │
    └─────┬─────┘
          │
    [Forward Pass]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | 如何计算整个模型？ | 🔄 流程分析 | 顺序执行 | Forward Pass |
| 底层 | 单层如何计算 | 📉 瓶颈分析 | 算子融合 | Kernel Fusion |
| 底层 | 精度与速度 | ⚖️ 对比分析 | 混合精度 | AMP |
| 底层 | 计算图 | 📉 瓶颈分析 | 图优化 | Graph Optimization |

**Forward Pass = 算子 + 融合 + 精度**

### 9.6 Quantization (28) - 量化 📉⚖️

```
    [FP32 → INT8] 📉 瓶颈分析
          │
    ┌─────┴─────┐
    │           │
 [权重量化] [激活量化]
    │           │
    ▼           ▼
 [Per-Channel] [Dynamic]
    │           │
    └─────┬─────┘
          │
    [Quantization]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | 如何减少内存？ | 📉 瓶颈分析 | 降低精度 | Quantization |
| 底层 | 权重怎么量化 | 📉 瓶颈分析 | 标度缩放 | Scale |
| 底层 | 量化方案 | ⚖️ 对比分析 | INT8/INT4 | 精度选择 |
| 底层 | 如何减少误差 | ⚖️ 对比分析 | 校准 | Calibration |

**Quantization = Scale + 方案 + 校准**

### 9.7 Speculative Decoding (30) - 推测解码 🎯

```
    [自回归 → 推测] 🎯 需求驱动
          │
    ┌─────┴─────┐
    │           │
 [起草] [验证]
    │           │
    ▼           ▼
 [Proposer] [Verifier]
    │           │
    └─────┬─────┘
          │
    [Speculative Decoding]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | 如何加速自回归？ | 🎯 需求驱动 | 推测 | Speculative Decoding |
| 底层 | 如何起草？ | 🎯 需求驱动 | N-gram/小模型 | Proposer |
| 底层 | 如何验证？ | 🔄 流程分析 | 大模型验证 | Verifier |
| 底层 | 如何接受？ | ⚖️ 对比分析 | 拒绝采样 | Acceptance |

**Speculative Decoding = Proposer + Verifier + Acceptance**

### 9.8 Prefix Caching (38) - 前缀缓存 🎯⚖️

```
    [相同前缀 → 缓存] 🎯 需求驱动
          │
    ┌─────┴─────┐
    │           │
 [如何存储] [如何查找]
    │           │
    ▼           ▼
 [Key-Value] [哈希]
    │           │
    └─────┬─────┘
          │
    [Prefix Cache]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | 相同前缀能复用？ | 🎯 需求驱动 | 缓存 | Prefix Caching |
| 底层 | 如何标识前缀 | 📉 瓶颈分析 | 哈希 | Hash |
| 底层 | 如何存储 | 🔄 流程分析 | KV Cache | Storage |
| 底层 | 缓存满了？ | ⚖️ 对比分析 | 驱逐 | Eviction Policy |

**Prefix Caching = Hash + Storage + Eviction**

### 9.9 Scheduler (35) - 调度器 ⚖️🎯

```
    [多请求 → 执行] ⚖️ 对比分析
          │
    ┌─────┴─────┐
    │           │
 [优先级] [资源分配]
    │           │
    ▼           ▼
 [FCFS/Priority] [GPU分配]
    │           │
    └─────┬─────┘
          │
    [Scheduler]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | 请求先后的顺序？ | ⚖️ 对比分析 | 调度 | Scheduler |
| 底层 | 调度策略 | ⚖️ 对比分析 | FCFS/优先级 | Policy |
| 底层 | GPU不够 | 📉 瓶颈分析 | 抢占 | Preemption |
| 底层 | 如何选择 | ⚖️ 对比分析 | 亲和度 | Affinity |

**Scheduler = Policy + Preemption + Affinity**

### 9.10 Continuous Batching (34) - 动态批处理 ⚖️🎯

```
    [请求 → 批] ⚖️ 对比分析
          │
    ┌─────┴─────┐
    │           │
 [静态批] [动态批]
    │           │
    ▼           ▼
 [Fixed Size] [Marquez]
    │           │
    └─────┬─────┘
          │
    [Continuous Batching]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | 多请求一起算？ | ⚖️ 对比分析 | 批处理 | Batching |
| 底层 | 批大小固定？ | 📉 瓶颈分析 | 静态批 | Static Batch |
| 底层 | 动态加入？ | 🎯 需求驱动 | 动态批 | Continuous Batching |
| 底层 | 何时换出 | ⚖️ 对比分析 | 演化算法 | Policy |

**Batching = Static/Dynamic + Policy**

### 9.11 Quantization 方案细分 📉⚖️

```
    [量化方案] 📉 瓶颈分析
          │
    ┌─────┴─────┐
    │           │
 [训练后量化] [训练中量化]
    │           │
    ▼           ▼
 [GPTQ/AWQ] [QAT]
    │           │
    └─────┬─────┘
          │
    [Quantization]
```

| 方案 | 发现方法 | 描述 | 特点 |
|------|----------|------|------|
| INT8 | 📉 瓶颈分析 | 8位整数 | 2x压缩，精度损失小 |
| INT4 | 📉 瓶颈分析 | 4位整数 | 4x压缩，需要校准 |
| GPTQ | ⚖️ 对比分析 | 训练后量化 | 逐通道校准 |
| AWQ | ⚖️ 对比分析 | 激活感知量化 | 逐token校准 |
| SQ | ⚖️ 对比分析 | 标度量化 | 简单有效 |

### 9.12 Distributed (50) - 分布式 🎯📉

```
    [单卡 → 多卡] 🎯 需求驱动
          │
    ┌─────┴─────┐
    │           │
 [数据并行] [模型并行]
    │           │
    ▼           ▼
 [DP] [TP/PP/EP]
    │           │
    └─────┬─────┘
          │
    [Distributed]
```

| 层级 | 问题 | 发现方法 | 解决 | 概念 |
|------|------|----------|------|------|
| 表层 | 单卡不够？ | 🎯 需求驱动 | 多卡 | Distributed |
| 底层 | 多卡如何分工 | 🎯 需求驱动 | TP/PP/EP | 并行策略 |
| 底层 | 参数如何分 | 📉 瓶颈分析 | 层切分 | Sharding |
| 底层 | 通信如何优化 | 📉 瓶颈分析 | NCCL | Communication |

**Distributed = 并行策略 + Sharding + NCCL**

---

## 10. 完整推导链总结

```
根：核心循环
  │
  ├── 编码 → Tokenizer
  │         │
  │         └── 分词模式 + 词表 + Byte
  │
  ├── 计算 → Transformer
  │         │
  │         └── Multi-Head + FFN + Residual + LN
  │                │
  │                ├── Attention → FlashAttention → Tiling + Online + Recompute
  │                │
  │                └── KV Cache → PagedAttention → Block + Table + Ref
  │
  ├── 采样 → Sampler
  │         │
  │         └── Softmax + Temp + Top-K/P
  │
  ├── 调度 → Scheduler
  │         │
  │         └── Policy + Preemption
  │                │
  │                └── Batching → Continuous → Dynamic
  │
  └── 优化 → Quantization
            │
            └── Scale + 方案 + 校准

            → Speculative Decoding
            │
            └── Proposer + Verifier + Acceptance
```

**这就是完整的50概念推导树！**

---

## 6. 核心公式

```
根 → 环节问题 → 分叉问题 → 概念
         (一因多果)
```

**关键洞察**：
- 一个父问题可能有**多个子问题**（分叉）
- 一个概念可能需要**多个问题**来解决（组合）
- 这不是链，是**网络**

---

## 7. 推导验证

### 从概念倒推

| 概念 | 需要解决什么问题？ | 属于哪个环节？ | 根是什么？ |
|------|-------------------|----------------|------------|
| PagedAttention | 显存碎片化 | 计算 | 核心循环 |
| Scheduler | 多请求处理 | 调度 | 核心循环 |
| FlashAttention | 计算量太大 | 计算 | 核心循环 |
| Speculative Decoding | 自回归太慢 | 优化 | 核心循环 |

### 从根出发

```
自回归生成循环
    ↓
分叉：编码/计算/采样/调度/优化
    ↓
每个分叉继续分叉
    ↓
最终概念
```

---

## 8. 记住这棵树

```
[根] 核心循环
  ├── [编码] → Tokenizer → Embedding
  ├── [计算] → Transformer → Attention
  │            ├── FlashAttention
  │            ├── PagedAttention
  │            └── Streaming
  ├── [采样] → Logits → Sampler
  ├── [调度] → Batching → Scheduler
  │            ├── Prefill/Decode
  │            └── Prefix Cache
  └── [优化] → Quantization
              → Speculative Decoding
              → Distributed
```

**50个概念，都是从这一棵树长出来的！**

---

## 11. 深层推导（子文档）

部分概念已拆分为独立文档，深入理解请查看：

| 主题 | 详细文档 | 核心公式 |
|------|---------|----------|
| 分词器 | [vllm-tree/tokenizer.md](./vllm-tree/tokenizer.md) | BPE → WordPiece → SentencePiece |
| 词嵌入 | [vllm-tree/embedding.md](./vllm-tree/embedding.md) | 查表 + 位置编码 |
| FlashAttention | [vllm-tree/flash-attention.md](./vllm-tree/flash-attention.md) | Tiling + Online + Recompute |
| PagedAttention | [vllm-tree/paged-attention.md](./vllm-tree/paged-attention.md) | Block + Table + Ref |
| 采样器 | [vllm-tree/sampler.md](./vllm-tree/sampler.md) | Softmax + Temp + Top-K/P |
| 量化 | [vllm-tree/quantization.md](./vllm-tree/quantization.md) | Scale + 方案 + 校准 |
| 推测解码 | [vllm-tree/speculative-decoding.md](./vllm-tree/speculative-decoding.md) | Proposer + Verifier + Acceptance |
| 前缀缓存 | [vllm-tree/prefix-caching.md](./vllm-tree/prefix-caching.md) | Hash + Storage + Eviction |
| 调度器 | [vllm-tree/scheduler.md](./vllm-tree/scheduler.md) | Policy + Preemption + Affinity |
| 批处理 | [vllm-tree/batching.md](./vllm-tree/batching.md) | Static/Dynamic + Policy |
| 分布式 | [vllm-tree/distributed.md](./vllm-tree/distributed.md) | TP/PP/EP + Sharding + NCCL |

---

## 12. 问题发现方法论

**这些问题是 如何被发现的？**

### 方法1：流程分析法

从核心循环的每个环节发现问题：

```
根循环：编码 → 计算 → 采样 → 判断
           │       │       │
           ▼       ▼       ▼
      输入处理？ 计算复杂？ 采样策略？
```

### 方法2：瓶颈分析法

从性能瓶颈倒推问题：

```
发现：慢
    │
    为什么慢？→ O(N²)复杂度
    │       │
    为什么是N²？→ 需要QKᵀ矩阵
    │           │
    解决：FlashAttention
```

### 方法3：对比分析法

从"理想vs现状"找差距：

```
理想：高吞吐 + 低延迟 + 低显存
        │        │        │
        ▼        ▼        ▼
    差距：批处理  差距：串行  差距：碎片化
        │        │        │
        ▼        ▼        ▼
    Scheduler   PagedAttention
```

### 方法4：需求驱动法

从用户需求追溯问题：

```
用户需求：更长上下文
    │
    需要处理更长序列 → 显存爆炸
    │           │
    解决：分页管理
            │
            PagedAttention
```

**问题发现 = 多维度交叉验证**

详细见：[vllm-tree/问题发现方法论.md](./vllm-tree/问题发现方法论.md)