# vllm-kv-cache-manager(4) — KV Cache 管理子系统

## NAME

vllm-kv-cache-manager — LLM 推理中 KV Cache 的分页分配、前缀缓存与跨请求复用子系统

## SYNOPSIS

#include <vllm/v1/core/kv_cache_manager.py>

class KVCacheManager:  
block_pool: BlockPool  
coordinator: KVCacheCoordinator  
enable_caching: bool

## DESCRIPTION

Transformer 自回归推理中，每生成一个 token 需与所有历史 token 的 Key/Value  
张量做 attention 运算。若重算则复杂度 O(L)，缓存则 O(1)。KV Cache 由此而生。

但其内存管理面临三重挑战：

- 请求长度动态变化，预分配挫伤显存利用率
- 多请求并发调度，物理内存碎片阻碍分配
- 跨请求前缀高度重合，重复计算浪费吞吐

vllm-kv-cache-manager 以分页元语统一解决上述问题，向上服务于 Scheduler，  
向下驱动 GPU 物理内存。其设计灵感源自 OS 虚拟内存管理。

## CORE INFERENCE CHAIN

### Problem 1: 内存碎片与静态预分配

传统方案为每请求预留 max_model_len 的连续 KV 内存，造成：  
内部碎片：实际生成长度 << 预分配长度，空闲空间无法回收  
外部碎片：分配-释放循环在连续内存中产生不连续孔洞

#### 解决: PagedAttention — 固定大小块的离散分配

(a) Block Partitioning  
将 KV Cache 以 block_size (typ. 16–256 tokens) 切分为固定大小逻辑块:

N_blocks = ceil(total_tokens / block_size)

每块物理内存占用:

block_bytes = 2 × block_size × num_kv_heads × head_dim × dtype_bytes

(b) Block Table Indirection  
逻辑块序号 → 物理块地址，通过 block_table 映射:

physical_id = block_table[logical_id]

物理块无需连续，任意空闲块可被分配给任一逻辑位置。  
每请求仅维护 block_table (O(N_blocks) 整型数组) 而非连续 Tensor。

(c) On-Demand Allocation  
仅当 token 填满当前块时向 BlockPool 申请新块，而非预分配全长。  
内部碎片上限为 1 个块 (< block_size tokens)。[-28](https://developers.redhat.com/articles/2025/07/24/how-pagedattention-resolves-memory-waste-llm-systems)

=> 回收已释放块回 pool，消除传统方案 ~60% 的内部碎片与严重外部碎片。[-28](https://developers.redhat.com/articles/2025/07/24/how-pagedattention-resolves-memory-waste-llm-systems)

#### 新问题: 跨请求前缀重复计算

多个请求常共享相同的系统 prompt 或 few-shot 前缀。请求间独立分配物理块  
意味着共享前缀被各请求独立存储、独立计算，浪费显存与算力。

### Problem 2: 跨请求前缀重复

请求 B 与请求 A 共享前缀 [tok0..tok47]，B 需重算整个 prefix。  
显存占用: 相同 KV 数据存储 N 份 (N = 并发请求数)。  
计算开销: 共享前缀 prefill 阶段 GPU 做无用功。

#### 解决: Automatic Prefix Caching (APC) — Hash-Logic-Physical 三层映射

(a) Block Fingerprint  
每逻辑块以其前缀 tokens + 块内 tokens 唯一定义:

block_hash = H(token_prefix ∥ token_block)

其中 H 为 SHA-256 (实践可用更轻量散列)。[-13](https://docs.vllm.ai/en/v0.6.3/automatic_prefix_caching/details.html)

(b) Global Hash Table  
维护物理块到 hash 的全局映射:

cached_blocks: Dict[BlockHash, KVCacheBlock]

alloc 时先查 hash_table: 命中则复用物理块 (ref_cnt++)，未命中才分配新块。  
free 时仅 ref_cnt--；当 ref_cnt==0 且块已满时，块进入 Cache 池 (可被 evict)  
而非直接回收。[-13](https://docs.vllm.ai/en/v0.6.3/automatic_prefix_caching/details.html)

(c) Copy-on-Write Fork  
请求 fork (如 beam search 分支) 时，复制 block_table 而非物理块。  
物理块共享，ref_cnt 递增；仅在写入时触发 COW 分配新块。

=> 共享前缀的物理存储从 O(N) 降至 O(1)，prefill 计算省去，首 token 延迟显著降低。  
内存效率提升正比于请求间的 token 重合度。[-14](https://docs.vllm.ai/en/v0.8.0/_sources/design/automatic_prefix_caching.md)

#### 新问题: 全局哈希表需驱逐策略

当 N_free_blocks → 0，新请求无法分配。哪些 cached 块应被驱逐以腾出空间？

### Problem 3: Cache Eviction Under Memory Pressure

Cached 块 (ref_cnt==0 且 _block_hash 已设置) 占用显存，  
必须选择驱逐受害者以最大化 cache 命中率。

#### 解决: 三级优先级 LRU Eviction Policy

(a) Priority Ladder

level 1: 驱逐 ref_cnt == 0 的 cached 块  
(被任何活跃请求引用的块不可驱逐)  
level 2: 在 L1 候选集中，优先驱逐 LRU (least recently used)  
level 3: 若最后访问时间相同，驱逐最长前缀末端块 (block_index 最大者)

该策略等价于 RadixAttention 的 refcount-zero + LRU leaf 策略，但实现在  
扁平哈希表而非前缀树之上，扩展性更优。[-13](https://docs.vllm.ai/en/v0.6.3/automatic_prefix_caching/details.html)

(b) BlockPool 数据结构  
free_block_queue: 双向链表 O(1) 任意位置移除  
cached_block_hash_to_block: Dict[BlockHash, KVCacheBlock]  
驱逐操作: 从队尾取块 → 清空数据 → 复用块索引给新 token

=> 在有限显存内最大化 cache 命中率，阻塞仅发生在块彻底耗尽时。

#### 新问题: 异构注意力模式下的统一管理

现代模型 (如 MoE hybrid、Phi-3、Gemma) 混合使用 full-attention、sliding-window、  
cross-attention 等多种模式。不同模式有不同块分配与回收策略。

### Problem 4: 多 KV Cache 组的协调

Single KV Manager 难以表达多种 attention pattern 的交织与独立分配需求。

#### 解决: KVCacheCoordinator — 多组管理器协调

(a) 架构  
KVCacheManager (统一接口)  
└── KVCacheCoordinator (策略派发)  
├── FullAttentionManager: 全块分配，无回收  
├── SlidingWindowManager: 仅保留窗口内块，回收旧块  
├── ChunkedLocalAttentionManager: chunk 内保留，外回收 + null-fill  
└── CrossAttentionManager: encoder 长度决定静态分配[-8](https://deepwiki.com/vllm-project/vllm/3.4-kv-cache-management)

(b) 协调逻辑  
allocate_slots(req_tokens):  
for each group_manager in coordinator:

# 各 group 独立分配，但共享同一 BlockPool

group_blocks[i] = manager.allocate(req_tokens)  
return merged KVCacheBlocks

KVCacheBlocks: blocks[i][j] 表示第 i 个 cache group 的第 j 个逻辑块。[-7](https://docs.vllm.ai/en/v0.13.0/api/vllm/v1/core/kv_cache_manager/#vllm.v1.core.kv_cache_manager.KVCacheBlocks)

=> 异构模型组件可按需选择最适合的分配策略，框架无需为每种模型硬编码特殊逻辑。

#### 新问题: 显存彻底告罄时的请求级响应

当所有块 (含 evictable) 耗尽，新请求无法继续。系统有两种选择: 拒绝请求  
或牺牲已在运行的请求。

### Problem 5: 调度级 Preemption — 空块耗尽时的请求抢占

BlockPool free_queue 为空 → allocate_slots() 失败 → Scheduler 需决策。

#### 解决: 请求级 Swap/Recompute Preemption

(a) Preemption 触发  
条件: Scheduler 检测到 KV cache 余量无法满足当前 batch 的最低 token 需求。  
动作: 选择 1 个或多个低优先级序列，将其 KV cache 块回收:

for victim_seq in selected_victims:  
KVCacheManager.free(victim_seq)   # ref_cnt-- → 可能 → cached_state

(b) 被抢占请求恢复  
当资源再次可用时，序列从 preempted 状态唤醒:-

- 若 token 仍在 context 内 (未超过 max_model_len)，  
从 prompt 起点重新 prefill，重新走 hash_lookup → allocate 流程。
- KV block 若未被驱逐 (仍在 cache)，直接复用，实现零重算恢复。
- 否则需完整重新计算 prefill。

=> 工作守恒: 吞吐量在资源紧张时仍保持最优，延迟换吞吐。

#### 新问题: 极高并发下的显存层级利用

单 GPU 显存有限，千万 token 上下文长度场景无法容纳全部 KV Cache。

### Problem 6: GPU 显存上界与 Swap

即使分页消除内部碎片，总显存仍受物理 VRAM 限制。频繁 preemption 影响延迟 SLA。

#### 解决: CPU Block Swap 作为二级存储

(a) 双级 BlockPool  
GPU blocks: num_gpu_blocks = floor(gpu_memory × utilization / block_bytes)  
CPU blocks: num_cpu_blocks 供 swap 使用[-38](https://docs.vllm.ai/en/v0.10.0/api/vllm/core/block_manager.html)

(b) Swap-in / Swap-out  
swap_out(blocks):  
for blk in blocks:  
copy D2H (GPU_block → CPU_block)  
mark GPU_block as free (加入 free_queue)  
swap_in(blocks):  
allocate free GPU_blocks  
copy H2D (CPU_block → GPU_block)  
restore mapping

(c) GpuMemoryAllocator 预分配  
引擎启动时按 gpu_memory_utilization (typ. 0.90) 比例预分配所有 GPU 块，  
避免运行时 cudaMalloc 开销。-

=> 长尾请求可暂存 CPU，短交互保持 GPU 低延迟，形成冷热分层。

## ARCHITECTURE SUMMARY

Scheduler (request scheduling)  
│  
├── get_computed_blocks(token_ids) → [KVCacheBlock]  
├── allocate_slots(req, num_new_tokens) → KVCacheBlocks  
├── free(seq_id)  
└── cache_blocks(blocks) [for full blocks]  
│  
v  
KVCacheManager  ── 统一门面，聚合 coordinator + block_pool  
│  
├── KVCacheCoordinator  ── 多 group 协调  
│     ├── FullAttentionManager  
│     ├── SlidingWindowManager  
│     ├── ChunkedLocalAttentionManager  
│     └── CrossAttentionManager  
│  
└── BlockPool  ── 物理块池，实现 alloc/free/cache/evict  
├── free_block_queue  (双向链表, O(1) 中段移除)  
├── cached_block_hash_to_block: Dict[Hash, KVCacheBlock]  
└── KVCacheBlock  
├── block_id: int  
├── ref_cnt: int         (>0 = 使用中；=0 = 空闲/cached)  
├── _block_hash: Hash    (已满块才设置)  
└── last_accessed: ts

Block Allocation Flow:

1. Scheduler 调用 allocate_slots(seq, n)
2. KVCacheCoordinator 遍历所有 group
3. SingleTypeKVCacheManager.allocate():  
for each new logical block:  
hash = compute_hash(prefix ∥ block_tokens)  
cached = block_pool.get_cached_block(hash)  
if cached: cached.ref_cnt++; return cached  
else:      blk = free_queue.popleft()  
blk.ref_cnt = 1; return blk
4. 返回 KVCacheBlocks，Scheduler 据此组装 block_table 传给 GPU kernel

## KEY INVARIANTS

Σ(ref_cnt_i × block_bytes_i) ≤ total_gpu_memory_allocated  
∀ block, if ref_cnt > 0 ⇒ block ∉ free_queue  
∀ cached block, if ref_cnt > 0 ⇒ block_hash exists  
logial_blocks(seq) ∝ seq_len / block_size  
physical_blocks ≤ num_gpu_blocks (hard ceiling)

## FILES

vllm/v1/core/kv_cache_manager.py         主管理器  
vllm/v1/core/single_type_kv_cache_manager.py  单 group 策略  
vllm/core/block_manager.py                块管理器 (v0 legacy)  
vllm/v1/core/block_pool.py                物理块池

## SEE ALSO

vllm(1), vllm-scheduler(4), pagedattention-kernel(4), vllm-automatic-prefix-caching(7)  
PagedAttention paper: arXiv 2309.06180  
vLLM APC design: vllm-project/vllm PR#3492

## BUGS

hash collision 理论存在但概率极低 (SHA-256)；实际部署中未见显著碰撞。  
CPU swap 延迟受 PCIe 带宽约束，intensive swapping 下吞吐退化。

## COLOPHON

vLLM v0.13.0+   2026-04-30   vllm-kv-cache-manager(4)