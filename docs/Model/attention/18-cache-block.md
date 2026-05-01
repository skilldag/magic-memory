```text
VLLM-CACHE-BLOCK(7)            vLLM 核心组件           VLLM-CACHE-BLOCK(7)

NAME
  vllm-cache-block — PagedAttention 中 KV 缓存分块管理的核心原理

DESCRIPTION
  本手册以问题→子概念/解决→新问题的层层推导方式，阐述 vLLM 推理框架
  中 **cache block** 的设计机理。读者应具备自回归语言模型、transformer
  注意力机制及 GPU 显存管理的基础知识。

  ● 问题 1：变长序列的 KV 缓存，预分配连续内存导致显存碎片化和利用率低下。
    如何以高弹性、低碎片的方式管理动态增长的 KV 缓存？
    ├─ 子概念：KV 缓存分块 (Cache Block)
    │  定义固定容量的 token 块作为最小存储单元。
    │  • 块大小 B (tokens/block)，典型值 16。
    │  • 单块包含模型所有层、所有注意力头的 K/V 张量。
    │  • 块内存量 = 2 × L × H × d × B × sizeof(dtype)
    │    L 层数，H 头数，d 头维度，dtype float16/bfloat16。
    ├─ 解决过程：逻辑到物理的分离与按需分配
    │  1. 将序列的 KV 缓存视作逻辑块序列 (logical block id)。
    │  2. 维护全局空闲物理块池。生成新 token 时，若当前逻辑块已满，
    │     从池中分配新物理块，追加到逻辑块链表尾部。
    │  3. 建立块表 (Block Table)，记录逻辑块到物理块的映射：
    │        physical_id = BlockTable[logical_id]
    │     物理块可离散分布于显存，彻底消除外部碎片。
    │  4. 序列结束后，其占用的物理块释放回空闲池。
    └─ 引出问题 2：注意力计算内核期望 K/V 张量在连续内存上，
       如何根据块表高效收集分散的物理块并完成矩阵运算？

  ● 问题 2：如何在非连续的、由块表定义的 KV 缓存上执行融合注意力？
    ├─ 子概念：PagedAttention 内核
    │  一个 GPU 算子，将 token 的块表作为输入，实时定位并加载 K/V 片段。
    ├─ 解决过程：
    │  1. 给定查询向量 q (shape: 1×d)，以及目标序列的块表 T 和序列长度 s。
    │  2. 内核遍历逻辑位置 i ∈ [0, s-1]：
    │        block_idx = i // B
    │        offset    = i %  B
    │        phys_base = T[block_idx] × block_bytes  （物理块基址）
    │        k_i = load(phys_base + offset × stride_k)
    │        v_i = load(phys_base + offset × stride_v)
    │  3. 对收集的 K、V 执行标准缩放点积注意力：
    │        attn = softmax(q @ K^T / √d)
    │        out  = attn @ V
    │     实际实现中融合 softmax 与掩码，逐块累加。
    └─ 引出问题 3：多序列并发（并行采样、beam search、共享前缀）时，
       如何避免相同前缀的 KV 重复存储，并高效管理物理块生命周期？

  ● 问题 3：如何使多个逻辑序列安全地共享物理块？
    ├─ 子概念：引用计数与 Copy‑on‑Write
    │  每个物理块维护一个引用计数 refcount，记录被多少逻辑块引用。
    │  共享为只读；当需要写入已共享的物理块时，触发块复制。
    ├─ 解决过程：
    │  1. 共享前缀：序列 A 和 B 共享前 N 个逻辑块，仅分配一份物理块，
    │     refcount = 2。各自生成新 token 时追加新的私有物理块，
    │     不会修改共享块（KV 缓存性质为追加写）。
    │  2. 写入冲突（罕见但存在）：若某序列必须修改共享块内容（如特殊
    │     的回退 / 修剪操作），执行 Copy‑on‑Write：
    │        a) 分配新物理块 P_new
    │        b) 将原物理块 P_old 内容复制至 P_new
    │        c) 更新该序列的块表指向 P_new
    │        d) P_old 引用减 1
    │  3. 回收：当 refcount 降至 0，物理块返回空闲池。
    │  物理块复用度 = (Σ 序列块数) / (Σ 分配物理块数)
    └─ 引出问题 4：块大小 B 如何影响显存利用率与计算效率？应如何选取？

  ● 问题 4：块大小 B 的决定论——内部碎片与元数据开销的权衡
    ├─ 子概念：内部碎片与块表开销
    │  • 内部碎片：最后一个逻辑块中未使用的 token 槽位。
    │    期望碎片 ≈ B/2 tokens / 序列 (当序列长度均匀分布)。
    │  • 块表内存：每个序列需 tot_blocks × sizeof(block_id) 字节。
    │    tot_blocks = L_max / B， L_max 为序列最大长度。
    ├─ 分析过程：
    │  1. 总浪费 W ≈ (内部碎片 KV 量) + (块表元数据量)
    │     W ≈ (B/2 × 2 × L × H × d × dtype) + (L × L_max / B × ptr_size)
    │  2. 对 W 关于 B 求极值，得到理论上界；实践中 B 过小会显著增加
    │     PagedAttention 内核循环迭代次数，削减吞吐。
    │  3. 经验最优 B 在 16–32 (token) 之间，vLLM 默认 16。
    └─ 引出问题 5：在多请求异构到达的场景下，如何实现物理块的快速分配、
       回收，并自动捕获前缀共享？

  ● 问题 5：块管理器如何支持高吞吐调度与前缀缓存？
    ├─ 子概念：块管理器与自动前缀缓存 (Prefix Caching)
    │  • 空闲块链表 (free list)：O(1) 分配与回收。
    │  • 哈希前缀匹配：对 token 序列的每个块计算内容哈希，建立
    │    hash → physical_block 的映射表。
    ├─ 解决过程：
    │  1. 新请求 prefill 阶段：
    │        for each logical block of prompt:
    │            h = hash(tokens[block_range])
    │            if h in prefix_cache:
    │                重用物理块，refcount++
    │            else:
    │                分配新块，计算 KV，插入哈希表
    │  2. 生成阶段：仅追加新块，复用任何可匹配的前缀块。
    │  3. 资源耗尽时，可抢占并交换 (swap) 不活跃序列的物理块到 CPU 内存，
    │     释放 GPU 空闲块，后续再换入。
    │  前缀缓存命中率 = 复用块数 / 总请求块数，理想情况下可大幅减少
    │  重复 prefill 计算。
    └─ 推导结束：cache block 通过分块映射、PagedAttention、共享与
      前缀缓存，实现了近乎零碎片的 KV 缓存管理，并将显存利用率提升至
       模型并行之外的又一关键维度。

SEE ALSO
  vllm(1), PagedAttention 论文《vLLM: Easy, Fast, and Cheap 
  LLM Serving with PagedAttention》, vLLM 源码 `cache_block.py`

vLLM 项目                         2024-05-01              VLLM-CACHE-BLOCK(7)
```