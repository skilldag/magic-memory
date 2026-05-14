```text
CONTINUOUS-BATCHING(7)         推理调度                  CONTINUOUS-BATCHING(7)

名称
       continuous-batching — 动态批次调度，最大化 GPU 利用率的 LLM 推理策略

概述
       Continuous Batching 以迭代为粒度动态组织批次，允许请求立即加入，
       已完成序列立即退出，并混合 prefill 与 decode 阶段，彻底消除
       传统 static batching 的 GPU 空闲气泡，将吞吐推至硬件极限。

描述
       本页沿问题链推导 Continuous Batching 的设计逻辑，从批次等待
       浪费出发，逐步构建迭代级调度、混合阶段与抢占机制。

问题 0 · 为何传统批次调度导致 GPU 严重闲置？
│
├── 背景：LLM 推理由多次模型前向传播组成，每次生成一个 token。
│       请求长度不一，生成步数各异。
│
├── Static Batching 流程：
│   - 收集一批请求，待全部到达后组 batch。
│   - 整批执行 prefill → 逐 token decode，直至 batch 内所有
│     序列均生成结束符或达最大长度。
│   - GPU 在批次生命周期中被独占，中间不插入新请求。
│
├── 浪费分析：
│   - 短板效应：批次耗时由最长序列（生成步数最多者）决定。
│     短序列完成后，其槽位空闲，GPU 仍为空槽位执行无效计算或 padding。
│   - 首 token 延迟：新请求必须等待当前整批处理完毕才能开始 prefill，
│     等待时间等于当前 batch 剩余最长生成长度。
│   - 利用率公式（简化）：
│        平均利用率 ≈ （seq_avg_len / seq_max_len）× padding_factor
│     序列长度差异越大，浪费越严重。中位序列可能仅用 30% 的时间。
│
└── 需求：能否让序列完成后立即退出、新请求立即加入，
    且执行粒度细到每次迭代（单一 forward）？

问题 1 · 如何在单次迭代中动态增删序列？
│
├── 核心理念：迭代级批次重构
│   - 将 batch 定义为一次模型前向的输入集合，不再是固定的一组请求。
│   - 每次迭代前，调度器根据当前状态构造 batch：
│        · 包含所有活跃 decode 序列（每条取最后 1 token）。
│        · 按显存和块配额加入一个或多个 prefill 请求的新序列。
│   - 任一序列生成了 EOS，则立即从下一迭代的 batch 中移除；
│     其占用的物理 KV 块（若采用 PagedAttention）当即释放。
│
├── 优势：
│   - 新请求 prefill 与正在 decode 的序列重叠执行，无需等待整批完成。
│   - GPU 持续满载：理想情况无任何迭代出现空闲槽位。
│   - 短序列即刻释放资源，长序列不阻塞后续任务。
│
├── 调度的数学模型（简述）：
│   设第 i 次迭代的 batch B_i 包含 D_i 条 decode 序列和 P_i 条
│   prefill 序列，受显存上限 M 约束：
│       M(B_i) = Σ_{seq∈B_i} 块数(seq) × 块大小 ≤ M_max
│   调度器最大化吞吐等价于在满足 M 约束下，每步选取最大可行的
│   D_i + P_i* （受 prefill 计算量权重调整）。
│
└── 新挑战：单次迭代内同时包含 prefill（计算密集长序列）和
    decode（访存密集单 token）可能造成计算不均，如何高效执行？

问题 2 · 如何混合 prefill 与 decode 阶段？
│
├── 分析：
│   - Prefill 需并行处理输入 prompt 的所有 token，产生大量
│     矩阵乘法（compute-bound）。
│   - Decode 每序列仅处理 1 个新 token，主要受 KV 缓存访问
│     带宽限制（memory-bound）。
│   - 二者直接混合会使 GPU 执行资源竞争，prefill 长序列可能
│     抢占总线导致 decode 延迟毛刺。
│
├── 子概念：Chunked Prefill（分块预填充）
│   - 将长 prompt 切分为固定尺寸的 chunk（如 512 token）。
│   - 每个迭代只处理一个 prefill chunk，剩余部分暂停，
│     类似被“抢占”的上下文。
│   - 该次迭代内，prefill chunk 与多 decode 序列一起计算。
│   - 优点：单次迭代的计算负载平滑可控，decode 的延迟保持
│     低且可预测，同时仍实现了 prefill 与 decode 并发。
│
├── 调度配合：
│   - 每次迭代前，调度器为每条 prefill 序列确定本次处理 token 数。
│   - 每完成一个 chunk，更新序列状态。prefill 全部完成后该序列
│     转为 decode 状态。
│   - GPU kernel 支持基于 block_table 统一寻址，使 prefill chunk
│     和 decode token 在 PagedAttention 核内同步计算。
│
└── 引出问题：如果显存耗尽，但仍有高优先级请求需立即服务，
    如何处理？

问题 3 · 如何在显存不足时维持服务并保证公平？
│
├── 子概念：序列抢占与换出（Preemption）
│   - 当新请求所需的物理 KV 块超过空闲池时，选择低优先级
│     或生成步数过多的序列，将其 KV 块从 GPU 显存换出到 CPU
│     内存（或直接丢弃并记录 token 序列）。
│   - 被换出序列的状态（block_table, 已生成 token 等）保留在
│     主机端，待资源宽裕时重新换入并继续解码。
│
├── 换入/换出策略：
│   - 换出：选取 victim 序列（如 FIFO 或基于优先级），释放其
│     物理页并转移数据。
│   - 换入：恢复块分配，可能需要重新运行 prefill 或从换出点
│     加载 KV 块。
│   - 该机制将 Continuous Batching 的弹性扩展至超订场景，
│     在过载时优雅降级而不是拒绝。
│
└── 闭环：迭代级重构 + 分块 prefill + 抢占换出，
    使 Continuous Batching 成为支持动态、高强度负载的完整
    调度框架，是现代 LLM 推理引擎（vLLM, TGI 等）的核心调度器。

参见
       vllm-arch(7), pageattention(7), flash-attention(7)

版本
       推理系统核心调度范式，随迭代级优化持续演进。

CONTINUOUS-BATCHING(7)                                     2026-05-02
```