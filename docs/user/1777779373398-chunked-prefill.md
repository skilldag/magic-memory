```text
CHUNKED-PREFILL(7)            推理调度组件              CHUNKED-PREFILL(7)

名称
       chunked-prefill — 将长提示的预填充切分为固定块，与解码流式混合
                        执行，消除延迟毛刺

概述
       Chunked Prefill 针对一次性 prefill 长序列引发的高延迟、批次阻塞
       问题，将 prompt 编码拆成可控大小的块。每个迭代仅处理一块，
       并可与 decode 序列混合调度，使首 token 延迟从 O(N²) 压至 O(N)
       恒定迭代内计算量，同时保持 GPU 持续满载。

描述
       本项目沿问题推导链层层展开，解释 chunked prefill 为何必要、
       如何切分、如何融入连续批处理以及如何处理状态一致性。

       问题 0 · 一次性 prefill 长提示为何危及服务质量？
       │
       ├── 标准 prefill 一次性处理整个 prompt，计算量 ∝ N² d。
       │   对长序列（如 32k），单次迭代耗时可达数百毫秒至秒级。
       │
       ├── 延迟后果：
       │   - 首 token 延迟（TTFT）巨大，用户感觉“卡住”。
       │   - 若采用连续批处理，一次 prefill 阻塞所有正在 decode
       │     的序列，形成计算尖峰（毛刺），其他请求的 token 生成
       │     被迫等待，破坏实时性。
       │   - 调度器只能等待 prefill 完成才能进行下一次重构，
       │     灵活性丧失。
       │
       ├── 公式表示（忽略 batch 维度）：
       │        T_prefill ≈ k · N² d   （k 为硬件系数）
       │        T_decode_iter  ≈ c · d² + m · L · d  （L 为已缓存长度）
       │   若 N ≫ 1，T_prefill ≫ T_decode_iter，二者不兼容。
       │
       └── 核心需求：将 prefill 计算量打散到多次迭代中，
           使其单次耗时与 decode 迭代可比，实现平滑调度。

       问题 1 · 如何将 prefill 分割为多个低延迟的前向步？
       │
       ├── 子概念：固定大小的 token 块（Chunk）
       │   - 定义块大小 C（如 512 或 1024 tokens）。
       │   - 对于长度为 N 的 prompt，划分为 ⌈N / C⌉ 个块。
       │         prompt_chunk_i = tokens[ i*C : min((i+1)*C, N) ]
       │
       ├── 逐块处理流程：
       │        初始：序列状态设为 prefill 正在进行，预分配逻辑块。
       │        第 1 次迭代：输入 chunk_0（前 C 个 token），执行因果
       │                     自注意力，将该 chunk 的 K/V 写入缓存页。
       │                     记录当前处理进度（consumed_tokens = C）。
       │        第 2 次迭代：输入 chunk_1（下一个 C token）。
       │                     注意力可关注已缓存的 chunk_0 的 KV（因果）。
       │                     追加 K/V 至缓存。
       │         ...
       │        最后 chunk 完成 → 状态转为 decode。
       │
       ├── 单块 prefill 计算量：
       │        T_chunk ≈ k · C² d    （C 为小块常数）
       │   可调整 C 使 T_chunk ≈ T_decode_iter，保证迭代时间均稳。
       │
       └── 因此，长 prompt 被分解为一系列块 prefill 迭代，
           总延迟虽因增加迭代次数而微增，但消除了巨型毛刺，
           首 token 在首个 chunk 完成后即可产生。

       问题 2 · 如何将分块 prefill 与 decode 在单次迭代内混合？
       │
       ├── 调度器在每次迭代构造 batch 时，可同时包含：
       │        · D 条 decode 序列（每序列 1 token）
       │        · P_chunk 条正在 prefill 的序列，每条处理一个 chunk
       │          （若有多条 prefill 请求，可各自取一个块）
       │
       ├── 混合执行的关键约束：max_tokens_per_iteration
       │   - 设每次迭代允许的最大 token 总数 T_max（由延迟 SLO 决定）。
       │   - 约束公式：
       │         D + Σ_{p} C_p ≤ T_max
       │     其中 C_p 是第 p 条 prefill 本块的实际 token 数。
       │   - 若有剩余 prefill chunk 无法容纳，则继续排队至下轮迭代。
       │
       ├── GPU 计算特点：
       │   - PagedAttention 核可通过 block_table 区分不同序列，
       │     在同一个 forward 中正确处理 decode token 与 prefill chunk
       │     的注意力计算。
       │   - prefill chunk 需要因果掩码（仅可见自身及之前块），
       │     decode token 需要看到自身及全部已缓存 K/V（含新写入块）。
       │     掩码构造在 kernel 内部根据序列类型适配。
       │
       └── 最终效果：GPU 每个迭代都满载 decode 与 prefill 负载，
           无空闲气泡，且迭代耗时严格受 T_max 上限控制。

       问题 3 · 分块 prefill 如何与 KV 缓存分页管理及前缀缓存协同？
       │
       ├── 物理页分配策略：
       │   - 在 prefill 开始时，根据总长度 N 预留逻辑块（ceiling(N/B)）。
       │   - 每处理完一个 chunk，将该 chunk 内的 K/V 按块粒度写入物理
       │     页（未满块可累积等待）。
       │   - 若该序列共享前缀（如系统提示），第一 chunk 匹配到缓存
       │     哈希后直接增加引用计数，零计算复用物理页。
       │
       ├── 写时复制（CoW）：
       │   - 若共享前缀因后续 token 分叉需修改（极少发生），在修改
       │     的 chunk 边界执行复制，保证隔离。
       │
       └── 内存回收：
           - 若 prefill 过程中显存耗尽，可抢占低优先级序列释放其页，
             本序列可换出或等待，chunked 的粒度使抢占损失仅限当前块，
             而非整个 prompt 的重新计算。
           - 完成 prefill 后，未使用的预留逻辑块可释放回池。

       问题 4 · Chunked Prefill 对延迟与吞吐的量化影响
       │
       ├── 首 token 延迟（TTFT）：
       │        TTFT_chunked ≈ T_chunk ≈ k · C² d
       │        TTFT_naive   ≈ k · N² d
       │   当 C=512, N=8192 时，加速比 ≈ (8192/512)² = 256×。
       │
       ├── 吞吐影响：
       │    chunked 增加少量迭代次数（多出 ceil(N/C)-1 次），但允许
       │    decode 序列插入这些额外迭代中，整体吞吐不降反升，因 GPU
       │    从空闲等待 prefill 变为全程满载。
       │
       └── 尾延迟优化：限制每次最大 token 数后，p99 延迟高度可控，
           满足在线服务的严格 SLO。

结论
       Chunked Prefill 将 prefill 从一个不可分割的高延迟阶段解构成
       一系列与 decode 尺寸一致的轻量迭代，消除了长提示的服务毛刺，
       并完美融入连续批处理调度框架。它与 PagedAttention、前缀缓存
       共同构成现代 LLM 推理系统平滑、高效、可预测的基石。

参见
       prefill(7), continuous-batching(7), pageattention(7),
       llm-iteration(7)

版本
       推理调度关键优化，广泛应用于 vLLM、TGI 等引擎。

CHUNKED-PREFILL(7)                                         2026-05-02
```