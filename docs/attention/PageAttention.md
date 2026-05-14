```text
PAGEATTENTION(7)            LLM 推理组件                  PAGEATTENTION(7)

名称
       pageattention — vLLM 中基于分页的 KV 缓存管理与注意力计算

概述
       PageAttention 是为解决自回归生成中 KV 缓存严重碎片化而设计的
       核心机制。它将缓存划分为固定大小的页，通过动态映射、定制 GPU 
       核及引用计数块管理，实现近 100% 的内存利用率与高效的多序列推理。

描述
       本页以问题推导链展开 PageAttention 的完整设计，每层解决前一层
       引发的挑战。

       问题 0 · KV 缓存预分配带来灾难性碎片
       │
       ├── 分析：每个序列为自回归生成缓存 K、V 矩阵，传统方案按最大
       │   长度 L_max 预留连续显存。
       │   内存利用率 ≈ L_avg / L_max
       │   例：L_max=8192, L_avg=512 → 利用率仅 6.25%
       │   原因：
       │   - 内部碎片：实际占用远小于容量。
       │   - 外部碎片：多序列交织分配/释放，空闲总量足够却无连续块。
       │
       ├── 子概念：分页存储抽象
       │   - 将 KV 缓存切分为固定大小的页（Page），每页含 B 个 token
       │     的 K,V 张量，形状 [B, num_heads, head_dim]。
       │   - 逻辑块序列 → 物理块映射表：
       │         block_table[seq_id][logical_idx] = physical_page_id
       │   - 全局物理页池：未使用的页可分配给任何序列的新逻辑位置。
       │   - 序列生成至逻辑块末尾时，按需从池中索取新物理页，若池空
       │     则可抢占低优先级序列并回收其页。
       │
       └── 成果
           - 外部碎片完全消失，仅余每个序列末尾 ≤ B-1 个 token 的
             内部浪费。B=16 时理论利用率可超 95%。
           - 显存支撑的并发序列数大幅提升，系统吞吐接近线性缩放。
           - 引发新问题：注意力计算需逐逻辑位置访问离散物理页，朴素
             gather+拼接 引入巨大开销。

       问题 1 · 如何高效计算离散页上的注意力？
       │
       ├── 朴素法：先按 block_table 将 K、V 拷贝到连续临时区，再调
       │   通用 MatMul。代价：多余显存拷贝 & 无法利用 FlashAttention
       │   分块优势。
       │
       ├── 子概念：PagedAttention 定制 GPU 核
       │   - 融合操作：一次核调用完成 查表 → 加载 K/V → 注意力计算。
       │   - 基于 FlashAttention 的分块递推：
       │     将 Q 沿序列维度分成小块 Q_block，对每个 Q_block 按逻辑
       │     顺序迭代序列的所有物理页，递推 safe softmax 的归一化常量。
       │
       │     核心递推（推导自标准 FlashAttention，此处适配分页）：
       │
       │     初始化： O = 0, l = 0, m = -∞
       │     for i in 0..num_logical_blocks-1:
       │         phys_id = block_table[i]
       │         K_i, V_i = load_page(phys_id)   # shape [B, H, D]
       │         S = Q_block · K_i^T / √d          # [Q_blk, B]
       │         m_new = max(m, rowmax(S))
       │         alpha = exp(m - m_new)
       │         beta  = exp(S - m_new)
       │         O = alpha * O + beta · V_i
       │         l = alpha * l + rowsum(beta)
       │         m = m_new
       │     输出： O = O / l
       │
       │   - 优势：完全不构造 N×N 注意力矩阵，仅按需加载当前物理页的
       │     K/V 块，全局访存量与标准 FlashAttention 持平，且零额外
       │     拷贝。B 常设为 16，兼顾 SRAM 利用率与内存碎片。
       │
       └── 引入新挑战：动态请求流中长度不一、随时到来，如何利用分页
          存储做出高效的块级调度，并避免重复存储公共前缀？

       问题 2 · 如何管理物理页的生命周期以实现最大吞吐？
       │
       ├── 子概念：块管理器 (Block Manager) 与写时复制
       │   - 引用计数：每个物理页维护计数，记录被多少逻辑序列共享。
       │   - 分配策略：prefill 新 token 需追加逻辑块时，若当前块未满
       │     → 直接使用；若满 → 分配新物理页并更新 block_table。
       │   - 自动前缀缓存：为每个逻辑块内容（token 序列）计算哈希，
       │     存入全局哈希表 (hash → 物理页列表)。新请求匹配 hash 后
       │     直接增加引用计数，零拷贝共享公共前缀的 K/V。
       │   - 写时复制 (Copy-on-Write)：某个共享页因生成分叉需要写入
       │     时，先复制该物理页，减原引用、加新引用，保证隔离。
       │   - 抢占与回收：显存不足时，依优先级回收整条序列的所有物理
       │     页（引用计数减一，归零则放回空闲池），并将序列状态换出，
       │     待恢复时重新借入。
       │
       ├── 与调度器的协同：连续批处理
       │   - 调度器每次迭代动态混合多个 prefill 与 decode 序列。
       │   - PagedAttention 核通过各自的 block_table 统一寻址，使
       │     混合批次的前向传播在单次核调用中正确完成。
       │   - 至此，分页存储、无碎片、高效核、智能共享与动态调度形成
       │     闭环，vLLM 得以在极低的显存浪费下将批次规模平稳推至硬件
       │     极限。
       │
       └── 归纳
           PageAttention = 分页存储抽象 + 定制递推核 + 引用计数前缀管理
           它不仅是算子优化，更是覆盖内存、计算、并发的完整子系统。

参见
       vllm(1), FlashAttention(3), 《Efficient Memory Management for
       Large Language Model Serving with PagedAttention》(SOSP'23)

版本
       自 vLLM 0.x 引入，为系统基石。

PAGEATTENTION(7)                                           2026-05-02
```