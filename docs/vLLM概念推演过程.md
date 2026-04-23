# vLLM 概念推导过程（树形结构）

> 以**最小循环为根**，**问题为边**，**概念为节点**的树形推导

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