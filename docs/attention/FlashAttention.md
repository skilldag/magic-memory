```text
TILING(7)                   并行计算优化                   TILING(7)

名称
       tiling — 将大规模算子分解为可装入快速缓存的独立小块，通过增量累加
               完成全局结果，是 IO 感知算法的核心组织范式

概述
       Tiling (分块) 是将无法一次容纳在高速内存 (SRAM) 中的矩阵或张量
       切割成在空间和时间上可管理的子块。每个子块完全在片上进行密集计算，
       结果通过递推或累加合并。它是解决“大矩阵，小缓存”矛盾的通用技术，
       也是 FlashAttention、PagedAttention、GEMM 等高性能核的数学骨架。

描述
       本手册以层层问题链展开 Tiling 的动机、机制、递推关联以及它在注意力
       中的具体实现，辅以实例与命名说明。

       问题 0 · 矩阵太大，装不进 SRAM，如何在不增加全局访存的前提下完成
               全矩阵运算？
       │
       ├── 硬件现实：GPU 的 SRAM 仅每 SM 拥有 ~100 KB，而典型矩阵规模
       │   远超此限。以注意力为例，Q, K, V ∈ R^{N×d}，若 N=8192, d=64，
       │   仅一个矩阵就占 4 MB。必须反复在 HBM 与 SRAM 间搬运数据。
       │
       ├── 核心概念：Tiling 将原矩阵沿一个或多个维度分割成固定大小的
       │   子矩阵 (Tile)，尺寸匹配 SRAM 容量。计算循环重新组织，使每次
       │   内循环只需加载一个或几个 Tile，完成 Tile 间的全部局部运算，
       │   最后将部分结果累加或递推合并。
       │
       ├── 命名来源：“Tile” 意为瓷砖。想象铺满整面墙的大图像，一次只能
       │   拿一块瓷砖绘制，最后拼成完整图案。Tiling 就是把大运算铺成小
       │   瓷砖，逐块完成。
       │
       └── 关键收益：
           · 每个 Tile 加载后可被高度复用（如在矩阵乘法中）。
           · 避免将完整的大矩阵中间结果写入 HBM，大幅压缩 IO。

       问题 1 · Tiling 如何应用于矩阵乘法？结合例子理解复用。
       │
       ├── 标准矩阵乘法 C = A × B，其中 A∈R^{M×K}, B∈R^{K×N}, C∈R^{M×N}。
       │   若 M,N,K 均很大，无法将完整 A,B 留在片上。
       │
       ├── 分块策略：将 A 按行分块，B 按列分块，K 维也分块。
       │       令块大小为 B_m, B_n, B_k。
       │       循环：for i in 0..M/B_m:
       │               for j in 0..N/B_n:
       │                   累加器 C_tile = zeros(B_m, B_n)  # 在片上
       │                   for k in 0..K/B_k:
       │                       加载 A_tile = A[i*B_m:(i+1)*B_m, k*B_k:(k+1)*B_k]
       │                       加载 B_tile = B[k*B_k:(k+1)*B_k, j*B_n:(j+1)*B_n]
       │                       C_tile += A_tile @ B_tile   # 片上矩阵乘法
       │                   写回 C_tile 到全局内存
       │
       ├── 数据复用分析：
       │   A_tile 在内循环 for k 中被每个 k 加载一次，但被用来与所有 B_tile 
       │   列块相乘。B_tile 也被重复使用于不同 A 行块。每次 HBM 加载的数据
       │   在 SRAM 中被多次消费，大幅提高算术强度。
       │
       └── 因此，Tiling 通过分解循环、引入片上累加，使原本 Θ(数据量) 的
           IO 降至 O(数据量/块尺寸)，实现带宽利用的最优化。

       问题 2 · Tiling 碰到需要全局归约的操作（如 Softmax）怎么办？
       │
       ├── 单纯累加对于乘加有效，但 Softmax 需要行级的最大值和指数和。
       │   这些是全局统计量，Tiling 使其不能独立计算每个块然后简单相加。
       │
       ├── 解决方案：在线递推 (Online Algorithm)。
       │   以在线 Softmax 为例 (详见 online-softmax(7))，我们维护每行的
       │   状态 (m, ℓ, O)，当处理一个新 K/V 块时，用递推更新这些状态。
       │   这种递推允许将全局归约嵌入分块循环中，使 Tiling 与 Softmax 共存。
       │
       ├── 核心原理：将全局归约转换为可增量更新的状态，状态尺寸极小（每行
       │   几个标量），因此可在 SRAM 中随 Tile 循环一起维护。
       │
       └── 这便产生 Tiling + Online Softmax 的组合，直接构成 FlashAttention
           的核心算法骨架。

       问题 3 · FlashAttention 中的 Tiling 如何工作？
       │
       ├── FlashAttention 将 Q 分割为块 Q_i，K, V 分割为块 K_j, V_j。
       │   每个 Q_i 块本身需要一个累加器 O_i 和对应的 softmax 状态。
       │
       ├── 算法结构（伪代码）：
       │       for i = 1 to N / B_q:
       │           加载 Q_i 到 SRAM，初始化 O_i=0, ℓ_i=0, m_i=-∞
       │           for j = 1 to N / B_kv:
       │               加载 K_j, V_j 到 SRAM
       │               S_ij = Q_i @ K_j^T / √d         # 片上计算
       │               (O_i, ℓ_i, m_i) = 在线更新( O_i, ℓ_i, m_i, S_ij, V_j )
       │               丢弃 S_ij, K_j, V_j
       │           写回 O_i 到 HBM
       │
       ├── Tiling 在此的角色：
       │   - 外循环 Q_i 将 N 序列维度划分，每个 Q_i 块复用所有 K,V 块。
       │   - 内循环 K_j, V_j 逐块流经，与 Q_i 完成全部交互后丢弃。
       │   - 所有中间 N×N 矩阵 S 完全不存在于 HBM，只以 Tile 形式在 SRAM 中
       │     被短暂产生并使用。
       │
       └── 因此，Tiling 重组了计算顺序，将 IO 从 O(N²) 降至 O(N² / B) 的
           量级 (B 为块大小因子)，使长序列注意力不再受带宽钳制。

       问题 4 · Tiling 与 PagedAttention 的关系是什么？
       │
       ├── PagedAttention 将 KV 缓存按固定大小的物理页组织，每个页本质
       │   上也是一个 Tile。它在内存管理层对 KV 块进行动态映射。
       │
       ├── 在执行时，PagedAttention 内核会遍历序列的逻辑块，通过页表获取
       │   物理块地址，并加载它们到 SRAM 进行计算。这些物理块的大小 (如 16)
       │   天然适合作为 Tiling 中的 K,V Tile。
       │
       └── 因此，Tiling 是计算层面的分块，PagedAttention 是存储层面的分页，
           两者在块粒度上协同，使得内存碎片消除与 IO 优化同时达成。

       记忆卡片：
       │
       ├── Tiling = 将大矩阵分解为可装入 SRAM 的小块 (Tile)
       ├── 作用：减少 HBM 访问，提高数据复用，克服内存墙
       ├── 循环重排：外循环固定某些块，内循环滑动另一维的块，片上累加
       ├── 结合在线算法：处理需要全局归约的算子 (如 Softmax)
       ├── 典型应用：FlashAttention 中的 Q/K/V 分块，矩阵乘法 GEMM
       └── 命名：瓷砖式铺满，一次一小块，拼合完成大计算

结论
       Tiling 是 IO 感知算法的基础设计模式。它通过将计算重整为对快速缓存
       友好的块序列，实现在有限 SRAM 下的高效执行。在 Transformer 注意力中，
       Tiling 与在线 Softmax 结合，消除了 O(N²) 中间矩阵的 HBM 读写，是
       FlashAttention 等高效内核的架构核心。理解 Tiling，就能理解为什么
       现代深度学习推理与训练可以处理超长序列。

参见
       flash-attention(7), online-softmax(7), sram(7), pageattention(7),
       matrix-multiplication(3)

版本
       源自经典高性能计算，深度学习中被 FlashAttention 等发扬。

TILING(7)                                                 2026-05-02
```