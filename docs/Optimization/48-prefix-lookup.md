```text
VLLM-PREFIX-LOOKUP(7)             vLLM Core              VLLM-PREFIX-LOOKUP(7)

NAME
   vllm-prefix-lookup — 基于哈希的自动前缀缓存，复用共享前缀的 KV 缓存

SYNOPSIS
   vLLM 在 prefill 阶段对 KV 缓存块进行内容哈希，当新请求与旧请求
   共享相同的前缀时，直接复用缓存的块，跳过重复计算。

DESCRIPTION
   LLM 推理中，长 system prompt 和多轮对话场景下，大量 token 在不同请求
   间重复计算，首 token 延迟 (TTFT) 极高。vLLM 的 Prefix Lookup 通过
   自动前缀缓存 (APC) 解决此问题。本手册以“问题→子概念/解决→新问题”
   的方式层进推导其原理。

   ● 问题 1: 多个请求（如相同 system prompt 的不同用户提问）共享同一段前
    缀，但每次 prefill 都要从头计算一遍 KV 缓存。能否把已算过的前缀
    存下来，新请求直接复用？
     ├─ 子概念: 自动前缀缓存 (Automatic Prefix Caching, APC)
     │  缓存已处理请求的 KV 缓存块，新请求共享相同前缀时直接重用，
     │  将昂贵的 prefill 转化为接近零开销的缓存查找[reference:0]。
     ├─ 解决过程: 前缀匹配 vs. 哈希匹配
     │  两种基本思路:
     │  · 精确前缀匹配: 扫描所有已有请求的 token 序列，找最长公共前缀。
     │    每次需要逐 token 对比，O(N) 复杂度随缓存量线性增长。
     │  · 哈希匹配: 对每个 KV 块计算内容哈希，一次 O(1) 查找命中。
     │  vLLM 选择后者，以块为粒度、以哈希为索引实现自动前缀匹配[reference:1]。
     └─ 引出问题 2: 块内 token 数量固定 (如 B=16)，单靠块内 token
       序列做哈希，不同位置但 token 序列相同的块会哈希碰撞。如何设计
       哈希键使每个块独一无二？

   ● 问题 2: 如何构造哈希键，使不同逻辑位置的 KV 缓存块可唯一标识且
     支持前缀语义？
     ├─ 子概念: 链式哈希 (Chained Hashing)
     │  哈希键 = (父块哈希, 本块 token 元组, 额外哈希)，
     │  将前缀链的上下文信息编码进每个块的哈希[reference:2]。
     ├─ 解决过程:
     │  1. 分块: 提示词按块大小 B 切分，未满一个块不缓存。
     │     示例 (B=4): "A gentle breeze stirred the leaves as children laughed"
     │     Block 1: [A, gentle, breeze, stirred]
     │     Block 2: [the, leaves, as, children]
     │     Block 3: [laughed]
     │  2. 计算哈希 (以 Block 3 为例):
     │        H₃ = hash( H₂, (laughed), extra_hashes )
     │     前两项编码了 "laughed" 及其前缀上下文[reference:3]。
     │  3. 额外哈希处理特殊场景:
     │     · LoRA: 附带 lora_int_id，区分不同 LoRA 适配器。
     │     · 多模态: 附带图像/音频的感知哈希，区分相同占位符下不同图像。
     │     · 多租户: 附带 cache_salt (随机盐)，防止时序侧信道攻击[reference:4]。
     │  4. KV 缓存 KV 块仅在完全填满时计算哈希并加入缓存。
     └─ 引出问题 3: 链式哈希使每个块的哈希依赖父块，请求到达时如何高效
       遍历并匹配共享前缀？

   ● 问题 3: 如何快速查找一个请求命中了多少已缓存的前缀块？
     ├─ 子概念: get_computed_blocks() — 基于哈希字典的前缀遍历
     │  调度器调用 KVCacheManager 的此方法，沿哈希链逐块查找已缓存块[reference:5]。
     ├─ 解决过程:
     │  1. 维护全局哈希字典: Dict[BlockHash, BlockId]，
     │     键为块的内容哈希，值为物理块 ID[reference:6]。
     │  2. 前缀遍历流程 (对请求的每个逻辑块依次执行):
     │        for i, block_tokens in enumerate(prefix_blocks):
     │            h = hash(parent_hash, block_tokens, extra_hashes)
     │            if h in _cached_blocks:
     │                parent_hash = h      # 命中，继续下一个块
     │            else:
     │                break                 # 未命中，停止遍历
     │  3. 返回: 命中的块序列 (Block IDs) 及剩余未命中 token。
     │  4. 命中的块直接挂载到新请求的块表，剩余 token 触发 prefill
     │     计算，新满块立即缓存供后续请求复用[reference:7]。
     └─ 引出问题 4: 显存有限，新块不断加入缓存，已缓存块何时淘汰？

   ● 问题 4: 在显存约束下，如何管理缓存块的分配与回收？
     ├─ 子概念: LRU 淘汰 + 引用计数 + 空闲链表
     │  · allocator: 统一管理 GPU 显存块。
     │  · evictor: LRU 策略淘汰引用计数为 0 的缓存块。
     │  · free queue: 双向链表维护空闲块，O(1) 分配/回收[reference:8]。
     ├─ 解决过程:
     │  1. 每个物理块维护 ref_cnt，表示被多少请求的块表引用。
     │  2. 请求完成时 ref_cnt 减 1；减至 0 时，块变成可淘汰但仍在哈希字典
     │     中 (弱缓存)，新请求仍可命中并复用。
     │  3. 显存紧张时:
     │        a) evictor 从 LRU 队列尾部 (最久未访问) 取块。
     │        b) 若 ref_cnt = 0，清除哈希记录并归还 free queue。
     │        c) 若 ref_cnt > 0，跳过 (仍被活跃请求占用)。
     │  4. 空闲块从 free queue 头部弹出分配；若弹出的块有哈希记录，
     │     先淘汰该记录再分配给新请求[reference:9]。
     └─ 引出问题 5: 哈希碰撞和多模态场景下占位符 token 都相同，如何
       保证缓存的正确性？

   ● 问题 5: 如何解决哈希碰撞与多模态占位符歧义？
     ├─ 子概念: SHA256 哈希 + 额外哈希 (extra_hashes)
     │  · 默认 Python hash → 高效但非加密安全，理论碰撞风险。
     │  · 生产多租户: 启用 SHA256，碰撞概率可忽略[reference:10]。
     │  · 多模态: 额外哈希携带图像嵌入的感知哈希，区分不同图像[reference:11]。
     ├─ 解决过程:
     │  1. 哈希算法配置 (vLLM v0.11+):
     │        默认: SHA256 (序列化: pickle)
     │        可选: SHA256_CBOR (跨语言确定性序列化)
     │        可选: XXHASH (128-bit, 更快但非加密)[reference:12]
     │  2. 多模态场景:
     │        · 图像被 tokenizer 替换为一系列占位符 token。
     │        · 额外哈希 = hash(image_embedding)，
     │          确保不同图像即使占位符相同也产生不同块哈希。
     │  3. 引用计数 + Copy-on-Write 保证共享块写入安全[reference:13]。
     │  因为前缀缓存是纯 KV 复用优化，不改变模型输出，被称为 "近乎免费的午餐"
     │  [reference:14]。
     └─ 推导结束: vLLM 的 prefix lookup 以块为粒度、链式哈希为索引、
       LRU+引用计数为淘汰策略、额外哈希为扩展机制，在 O(1) 查找时间内
       实现自动前缀缓存的透明复用，将高成本的 prefill 转化为廉价命中。

   核心公式总结:
   ┌──────────────────────────────────────────────────────────┐
   │ 块哈希公式                                                │
   │   H(block_i) = hash( H(block_{i-1}), tokens_i, extra )   │
   │   其中 H(block_0) = hash(∅, tokens_0, extra)             │
   │                                                          │
   │ 前缀命中查找复杂度: O(n_blocks) ≈ O(L_prefix / B)        │
   │ 哈希字典查找: O(1) / block                               │
   │                                                          │
   │ 缓存复用收益                                             │
   │   prefill_saved = n_cached_blocks × B × cost_per_token    │
   │   TTFT 降低 ∝ 缓存命中数                                 │
   │                                                          │
   │ 内存开销                                                 │
   │   哈希字典:  per_block ≈ 64B (指针 + 哈希值)             │
   │   引用计数:   per_block ≈ 4B                             │
   │   总开销 ≈ O(num_cached_blocks)，相对于 KV 块本身可忽略  │
   └──────────────────────────────────────────────────────────┘

SEE ALSO
   vllm(1), vllm-cache-block(7), PagedAttention 论文,
   vLLM 设计文档 `automatic_prefix_caching.md`,
   源码 `vllm/core/block/prefix_caching_block.py`

vLLM 项目                         2026-05-01            VLLM-PREFIX-LOOKUP(7)
```