# VLLM-CACHE-EVICTION(7) — vLLM 缓存驱逐核心原理

## 名称

**vllm-cache-eviction** — 基于分页注意力的 KV Cache 块级驱逐机制，用于在有限 GPU 显存下最大化前缀缓存复用。

## 描述

vLLM 将每个请求的 KV Cache 划分为固定大小的**物理块（Physical Token Block）**，块间允许非连续存储以消除内存碎片[-32](https://docs.vllm.ai/en/v0.7.0/design/automatic_prefix_caching.html)。当空闲块耗尽时，需将部分缓存块逐出（evict）以容纳新块，此即 **cache eviction**。本文档采用层层推导方式，从问题出发逐步展开其核心原理。

### 问题一：如何识别两请求共享了相同前缀？

子问题：若请求 B 的系统提示词与请求 A 完全相同，如何避免对 B 重复计算预填充？

│  **子概念：块哈希（Block Hash）**  
│  
├── def: 每个 KV 块由其 `(父块哈希, 本块 token 元组, 额外哈希)` 三元组唯一标识。  
│   └── 额外哈希包含：LoRA ID、多模态输入哈希、缓存盐值等[-29](https://docs.vllm.com.cn/en/latest/design/prefix_caching/#__codelineno-0-1)。  
│  
├── 构建过程:  
│   ├── 对请求按 block_size（默认 16）切分为逻辑块序列 [B₀, B₁, ..., Bₙ₋₁]  
│   ├── 递推计算哈希链:  
│   │   ├── h₀ = hash(tuple(tokens[B₀]))  
│   │   └── hᵢ = hash(hᵢ₋₁, tuple(tokens[Bᵢ]))  for i ≥ 1  
│   └── 得全序列块哈希向量 H = [h₀, h₁, ..., hₙ₋₁]  
│  
├── 关键等价性:  
│   └── hᵢ 相等 ⇒ 请求共享前缀 token[0..(i+1)·block_size)，即前 i+1 个逻辑块内容一致  
│  
└── ∴ 块哈希将任意长度前缀匹配问题归约为 O(1) 哈希查表。

│  **引出问题二**

### 问题二：已知块哈希，如何查找和复用已缓存块？

子问题：计算得到 H 后，需要一种全局数据结构来回答 `hᵢ in cache ?` 并返回对应物理块。

│  **子概念：全局哈希表（Global Hash Table）**  
│  
├── 结构定义:  
│   └── table: dict[BlockHash → PhysicalTokenBlock]  
│       每个条目存储: {content_hash, physical_block_id, ref_cnt, last_accessed, num_hashed_tokens}[-9](https://docs.vllm.ai/en/v0.10.2/api/vllm/core/evictor.html)  
│  
├── 操作流程（lookup 阶段）[-29](https://docs.vllm.com.cn/en/latest/design/prefix_caching/#__codelineno-0-1):  
│   ├── for hᵢ in H:  
│   │   ├── if hᵢ ∈ table ∧ block_is_ready:  
│   │   │   └── 命中计数 +1, 返回对应物理块  
│   │   └── else:  
│   │       └── break（一旦缺失，后续块必然不连续，终止查找）  
│   └── return (computed_blocks, num_hit)  
│  
├── 特殊处理:  
│   └── max_cache_hit_length = num_tokens - 1   # 必须保留至少 1 token 用于重计算 logits[-19](https://cumtchw.blog.csdn.net/article/details/158286408)  
│  
├── 引用计数（ref_cnt）:  
│   ├── 每增加一个请求引用该块: ref_cnt += 1  
│   ├── 每释放一个引用: ref_cnt -= 1  
│   └── ref_cnt = 0 ⇒ 块可被安全逐出（无人使用）[-32](https://docs.vllm.ai/en/v0.7.0/design/automatic_prefix_caching.html)  
│  
└── 通过哈希表，块之间独立于请求而存在，实现跨序列共享。

│  **引出问题三**

### 问题三：显存有限，当分配新块时缓存已满怎么办？

子问题：当 `free_blocks = 0` 且 `ref_cnt(block) > 0 for all cached blocks` 不可能，但现实中有大量 ref_cnt=0 的块。选择哪个逐出？

│  **子概念：逐出策略（Eviction Policy）**  
│  
├── 接口抽象:  
│   └── Evictor（抽象基类）: BlockAllocator 通过 make_evictor(policy) 工厂创建[-9](https://docs.vllm.ai/en/v0.10.2/api/vllm/core/evictor.html)-  
│  
├── 策略枚举:  
│   ├── EvictionPolicy.LRU      # 最近最少使用（默认策略）  
│   ├── EvictionPolicy.FIFO     # 先进先出  
│   └── EvictionPolicy.LFU      # 最不经常使用-  
│  
├── 默认 LRU 策略的三级优先级:  
│   ├── 1. ref_cnt == 0          # 仅淘汰无人引用的块  
│   ├── 2. last_accessed 最小    # 在无引用块中，选最久未访问者  
│   └── 3. 最长前缀末尾块         # 若访问时间相同，选所在前缀最长的末尾块[-32](https://docs.vllm.ai/en/v0.7.0/design/automatic_prefix_caching.html)  
│  
└── 决策树:  
if free > 0: allocate()  
else:  
candidates = {b | b.ref_cnt == 0}  
if candidates empty: return OOM  
victim = argmin_{b∈candidates} (b.last_accessed, -b.prefix_len)  
evict(victim); allocate()

│  **引出问题四**

### 问题四：LRU 如何感知访问模式变化？— 自适应替换缓存（ARC）

子问题：纯 LRU 无法区分“最近但仅用一次”与“频繁复用”的块，在多轮对话中可能误逐出热点前缀。

│  **子概念：ARC（Adaptive Replacement Cache）**[-8](https://docs.vllm.ai/en/v0.13.0/api/vllm/v1/kv_offload/arc_manager/#vllm.v1.kv_offload.arc_manager)  
│  
├── 四队列结构:  
│   ├── T1: 缓存仅访问一次的块（Recency 队列）  
│   ├── T2: 缓存访问≥2 次的块（Frequency 队列）  
│   ├── B1: Ghost 列表，记录最近从 T1 被逐出的块哈希  
│   └── B2: Ghost 列表，记录最近从 T2 被逐出的块哈希  
│  
├── 自适应目标:  
│   └── target_t1_size: 动态调整的 T1 容量目标（T1 + T2 = 总缓存大小 c）  
│  
├── 操作详解:  
│   ├── lookup(h): 顺序搜索 T1→T2，返回连续命中数  
│   │  
│   ├── touch(h): 按命中位置调整:  
│   │   ├── h ∈ T1: 从 T1 移至 T2（晋升为频繁项）  
│   │   ├── h ∈ T2: 移至 T2 队尾（MRU 位置）  
│   │   ├── h ∈ B1: target_t1_size += 1  （Ghost 命中说明追近访问更重要）  
│   │   └── h ∈ B2: target_t1_size -= 1  （Ghost 命中说明频繁访问更重要）  
│   │  
│   ├── evict(): 自适应选择逐出源:  
│   │   ├── if len(T1) ≥ target_t1_size: evict from T1 → add to B1  
│   │   └── else:                         evict from T2 → add to B2  
│   │  
│   └── insert(h): 新块始终插入 T1，若 h ∈ B1 ∪ B2 则从幽灵列表中移除[-2](https://docs.vllm.ai/en/latest/api/vllm/v1/kv_offload/cpu/policies/arc/)  
│  
├── 自适应机制有效性:  
│   ├── B1 命中 → 过去以最近访问模式为主，应扩大 T1  
│   ├── B2 命中 → 过去以频繁访问模式为主，应扩大 T2  
│   └── 幽灵列表以元数据开销换取信息增益，无需存储实际块内容[-8](https://docs.vllm.ai/en/v0.13.0/api/vllm/v1/kv_offload/arc_manager/#vllm.v1.kv_offload.arc_manager)  
│  
└── 均衡公式: 系统在受限容量 c 下自适应逼近 recency/frequency 最优权衡[-8](https://docs.vllm.ai/en/v0.13.0/api/vllm/v1/kv_offload/arc_manager/#vllm.v1.kv_offload.arc_manager)

### 问题五：eviction 在整个调度周期中何时触发？

子问题：逐出不是独立的后台线程，而是与 KV Cache 分配流程深度耦合。触发时机必须原子化保证一致性。

│  **子概念：惰性逐出（Lazy Eviction）**  
│  
├── 触发时机:  
│   └── 在 `prepare_store()` 中检测到 free_blocks < required[-39](https://docs.vllm.ai/en/v0.11.1/api/vllm/v1/kv_offload/lru_manager/)  
│  
├── 原子化保证:  
│   ├── 若可逐出块不足，返回 None（整体失败，无部分状态修改）[-39](https://docs.vllm.ai/en/v0.11.1/api/vllm/v1/kv_offload/lru_manager/)  
│   └── 成功则一次性批量逐出 → 分配 → 更新元数据  
│  
├── 调用链简图:  
│   Scheduler.schedule()  
│     → kv_cache_manager.get_computed_blocks()     # lookup/touch  
│     → kv_cache_manager.allocate_slots()           # prepare_store/evict  
│       → coordinator.find_longest_cache_hit()  
│       → evictor.evict() if needed  
│  
├── 时间线示意:  
│   t0: Request A 分配块 [0,1,2], free=97  
│   t1: Request B 命中块 [0,1], free=97  
│   t2: Request C 需 98 块 → free 不足 → evict 至少 1 块 → A 完成后 ref_cnt[2]=0 被逐出  
│  
└── ∴ eviction 永远在分配时“被动”触发，保持缓存一致性最高。

### 问题六：逐出后，原块对应 GPU 显存如何回收？

子问题：层级的块哈希/幽灵列表是对块引用的上层抽象，最终逐出需落实为物理内存释放。

│  **子概念：物理块回收**  
│  
├── 物理块结构: PhysicalTokenBlock { block_id, device, ref_cnt }  
│  
├── evict 的执行:  
│   ├── Evictor 层:  
│   │   └── 从哈希表/优先级队列中选出 victim_block_id  
│   ├── BlockAllocator 层:  
│   │   └── free(victim_block_id): 将物理块标记为 free  
│   └── GPU 显存层:  
│       └── 该物理块指向的显存区域进入 free_pool，供后续 alloc 复用  
│  
├── 回收不涉及数据擦除:  
│   └── 仅更新元数据状态为 free，GPU 显存内容被惰性覆写  
│  
└── 即逐出是将 `status[block_id]: used → free` 的元数据操作，与数据面解耦。

## 总结：完整 Eviction 推导链

```text
Q₁: 如何识别前缀共享？
    ↓ 子概念: 块哈希
    公式: hᵢ = hash(hᵢ₋₁, tokens[Bᵢ])

Q₂: 如何查找/复用已缓存块？
    ↓ 子概念: 全局哈希表
    规则: 顺序查找，一旦缺失即终止

Q₃: 缓存满时淘汰哪个块？
    ↓ 子概念: LRU 策略
    优先级: ref_cnt=0 ≻ min(last_accessed) ≻ max(prefix_len)

Q₄: 如何自适应访问模式变化？
    ↓ 子概念: ARC 四队列
    自适应: B1/B2 幽灵命中 → 调整 target_t1_size

Q₅: eviction 何时触发？
    ↓ 子概念: 惰性逐出
    时机: prepare_store() 时 free < required

Q₆: 逐出后物理内存如何回收？
    ↓ 子概念: 物理块回收
    操作: metadata.free() → GPU 显存进入 free_pool
```

## 参见

- `vllm-core-block-manager`(7) — PagedAttention 块管理器
- `vllm-prefix-caching`(7) — 自动前缀缓存设计
- `vllm-kv-offload`(7) — CPU offload 与分层缓存
- vLLM 设计文档: `docs/design/v1/prefix_caching.md`