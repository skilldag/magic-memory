```text
FLASH-ATTENTION(7)             GPU 算法库                  FLASH-ATTENTION(7)

名称
       flash-attention — 快速且内存高效的精确注意力算法

概述
       FlashAttention 通过 IO 感知的分块算法与在线 Softmax，将自注意力
       的 HBM 读写量削减约一个数量级，在严格保持数值等价的同时将计算
       搬至近核 SRAM，使 Transformer 支持更长序列。

描述
       自注意力的计算瓶颈常在于显存带宽而非算术强度。本页以层层推导
       展示 FlashAttention 如何从内存墙问题出发，逐步构建完整前向与反
       向传播方案。

问题 0 · 标准注意力为何受显存带宽钳制？
│
├── 计算背景：给定 Q, K, V ∈ R^{N×d}，d 为头维度（通常 64~128）
│        S = Q K^T / √d          [N×N]
│        P = softmax(S)          [N×N]
│        O = P V                 [N×d]
│
├── 瓶颈分析：
│   - 标准实现：将完整 S, P 矩阵写出到 HBM，再读回乘 V。
│   - HBM 读写量 Θ(N^2)，计算量 Θ(N^2·d)。
│   - d ≪ N 时（长序列），每字节数据只做少量乘加，成为 IO‑bound。
│   - GPU 层次：SRAM 带宽约为 HBM 的 10~20 倍，但容量仅 ~100 KB。
│
└── 受迫需求：能否切块后在 SRAM 中完成，避免 N×N 矩阵落盘到 HBM？

问题 1 · 如何分块计算 Softmax 以融合矩阵乘法？
│
├── 直觉：将 Q, K, V 切成允许放入 SRAM 的小块，逐块处理。
│   障碍：Softmax 依赖整行的指数和，无法直接分块独立计算。
│
├── 子概念：Online Safe Softmax（在线安全 Softmax）
│   - 对一行查询 q（写作向量），当 K 分块依次到达时，维护：
│         m      当前行的最大值（用于数值稳定）
│         ℓ      指数和的累计值
│         o      输出加权累加器
│
│   递推过程（初始 m = -∞, ℓ = 0, o = 0）：
│       对每一块 K_j, V_j：
│           s = q @ K_j^T / √d                 # 部分相似度
│           m_new = max(m, max(s))
│           放缩修正：
│               α = exp(m - m_new)            # 保护旧累加器
│               o = α · o
│               ℓ = α · ℓ
│           融入新块：
│               p = exp(s - m_new)             # 块内 softmax 分子
│               o = o + p @ V_j
│               ℓ = ℓ + sum(p)
│               m = m_new
│       最终输出： o = o / ℓ
│
│   - 数值等价：与显式写出 S 后全局 softmax 结果完全一致。
│   - 仅需 O(N·d) 的 SRAM 暂存，无需存储 N×N 矩阵。
│
├── 完整前向算法（双重 Tiling）：
│   - 外循环：将 Q 切成块 Q_i，每个 Q_i 独立对应整个序列。
│   - 内循环：对该 Q_i 迭代所有 K,V 块，执行如上递推。
│   - 仅需向 HBM 写入最终 O，Q 读一次，K,V 各读多次（重加载开销
│     仍远小于写入 N×N 方阵）。
│
│   HBM 访问量 从 Θ(N^2·d) 降至 Θ(N·d·M) 量级（M 为分块数），
│   长序列下实现 10× 加速。
│
└── 遗留问题：训练时反向传播需要中间激活 P 和 S，
    前向只留下了归一化统计量 (m, ℓ)。如何无 P 求导？

问题 2 · 反向传播如何避免存储大型中间矩阵？
│
├── 标准反向需要：
│        dS = dP ⊙ P          （需要 P）
│        dQ, dK, dV 由 dS 和 P 推出。
│
├── 子概念：激活重计算（Recomputation）
│   - 前向仅将 O 与每行的 (m, ℓ) 写回 HBM（总大小 Θ(Nd)）。
│   - 反向时：重新以一致的分块方式加载 Q,K,V，利用保存的 m, ℓ
│     在 SRAM 中逐块重算 P 和 S 的局部值，即时用于梯度流。
│
│   反向递推概要（简记）：
│       令 dO 已传播。对每行 i，按内循环顺序或逆序迭代块 j。
│       用保存的 m, ℓ 修正 dP 并计算：
│           dS_j = dP_j ⊙ exp(S_j - m) / ℓ
│       依链式法则得到 dQ_i, dK_j, dV_j 的局部贡献，累加。
│   - 全部操作在 SRAM 内完成，无额外 HBM 存储。
│
│   - 端到端训练 IO 同量级压缩，让大 N 训练成为可能。
│
└── 进化：FlashAttention-2 进一步优化了 GPU 线程块和 warp 的调度，
   减少非矩阵乘法计算，达到更高并行度；但核心思想不变——始终以
   IO 感知和 SRAM 分块 Softmax 为基石。

总结
       FlashAttention = Tiling + Online Safe Softmax + 反向重计算
       它将自注意力的 IO 模式从接近 N² 放缩变为 Nd 放缩，严格保持数值
       等价，成为现代 Transformer 训练与推理的基础组件。

参见
       vllm-arch(7), pageattention(7),
       《FlashAttention: Fast and Memory-Efficient Exact Attention
        with IO-Awareness》(NeurIPS 2022),
       《FlashAttention-2: Faster Attention with Better Parallelism
        and Work Partitioning》(2023)

版本
       FlashAttention 1/2/3 持续演进，此处覆盖基石版本。

FLASH-ATTENTION(7)                                         2026-05-02
```