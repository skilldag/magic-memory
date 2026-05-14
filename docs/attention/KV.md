```text
KV-CACHE(7)             Transformer 推理组件                 KV-CACHE(7)

名称
       kv-cache — 自回归解码中的键值缓存，消除历史 token 重复投影

概述
       KV 缓存通过在每一步存储已计算的 Key 和 Value 张量，避免为
       所有前缀 token 重新计算线性投影，将解码每步的投影计算量从
       O(t·d²) 降为 O(d²)，同时显存占用成为新的核心矛盾，驱动从
       架构压缩到系统分页的一系列优化。

描述
       本页以问题链形式推演 KV 缓存的动机、机制、瓶颈与优化方向，
       使用树状缩进与简洁公式。

问题 0 · 自回归生成为何必须缓存 Key 和 Value？
│
├── 背景：Transformer 解码层自注意力
│       给定输入序列 X ∈ R^{t×d}，经投影得：
│           Q = X W_Q,  K = X W_K,  V = X W_V
│       输出： O = softmax( Q K^T / √d ) V ，带上三角掩码。
│
├── 冗余观察：生成第 t+1 个 token 时，其注意力仅能看向
│       1..t 个前缀 token。这些前缀的 K_{1..t}, V_{1..t} 在
│       前 t 步已计算过且严格不变（无未来信息泄漏）。
│
├── 无缓存代价：每步重新为全部前缀计算 K,V 投影，
│       第 s 步计算量 ∝ s × (2 d²) （两次线性变换）。
│       总投影操作 ≈ ∑_{s=1}^N s × 2d² ∝ N² d² 。
│
├── 缓存解决方案：每步仅计算当前 token 的 q_s,k_s,v_s，
│       将 k_s, v_s 追加到固定大小的缓存张量中。
│       - 第 s 步投影量：恒为 2d²，与序列长度无关。
│       - 总投影量降至 N × 2d² = O(N d²) 。
│       - 注意力点积仍为 O(t·d)，此部分不变。
│   公式简化对比：
│       无缓存每步投影量： C_prj(s) = 2 d² s
│       有缓存每步投影量： C_prj(s) = 2 d²
│
└── 直接后果：缓存需在显存中保留不断增长的 K,V 序列，
    引出显存压力与碎片问题。

问题 1 · KV 缓存的物理存储与典型瓶颈
│
├── 物理布局（每层独立）：
│   Keys:   [batch, num_heads, max_seq_len, head_dim]
│   Values: [batch, num_heads, max_seq_len, head_dim]
│   通常预分配最大长度 max_seq_len 的连续张量，步骤 s 时
│   写入位置 s（或 s-1）。
│
├── 显存占用量（float16 示例）：
│   Mem = 2 × batch × num_layers × num_heads × max_seq_len × head_dim × 2 Bytes
│   以 Llama-7B (32 heads, 128 head_dim, 32 layers) 为例：
│       batch=1, max_len=2048 → 约 1.1 GB
│       batch=32, max_len=8192 → 约 57 GB （远超 7B 权重）
│
├── 两大浪费：
│   (1) 内部碎片：实际平均长度 L_avg ≪ max_len，
│       利用率 ≈ L_avg / max_len，常低于 50%。
│   (2) 预分配即占用全生命期，多请求并发时显存迅速耗尽。
│
└── 根源：连续、静态分配无法适应动态、不定长的序列。
    引出需求：能否压缩缓存本身大小？能否动态分配内存？

问题 2 · 如何通过模型架构削减 KV 缓存体积？
│
├── 子概念 A：Multi-Query Attention (MQA)
│   - 所有 query 头共享同一个 key 头和 value 头。
│       num_kv_heads = 1
│   - 缓存大小降为 MHA 的 1 / num_heads，访存压力陡降。
│   - 精简后模型容量轻度损失，需权衡。
│
├── 子概念 B：Grouped-Query Attention (GQA)
│   - query 头被划分为 G 组，每组共享一个 KV 头。
│       num_kv_heads = G，通常取 2~8。
│   - 缓存尺寸 = MHA × (G / num_heads)
│   - 质量接近 MHA，现为 Llama-2/3、Mistral 等主流标配。
│
├── 子概念 C：滑动窗口 (Sliding Window Attention)
│   - 仅缓存最近 W 个 token 的 K,V，丢弃更远的上下文。
│   - 缓存大小固定在 O(W)，与总长度 N 解耦。
│   - 适用于局部性强、或搭配全局层混合使用的模型。
│
└── 结构优化可缩减数倍缓存，但仍无法消除动态长度引发的
    内存碎片（外部碎片），且均需预分配最大窗口/长度。

问题 3 · 如何彻底消除 KV 缓存外部碎片？
│
├── 核心思路：将物理显存划分为固定大小页（Block），
│   序列的 KV 按逻辑顺序通过页表映射到物理页，
│   按需分配，释放时回收。 → 即 PagedAttention。
│   (详见 pageattention(7))
│
├── 伴随技术：
│   - KV 缓存量化：将 K/V 以 INT8 甚至 INT4 存储，
│     计算时反量化为 FP16，体积压缩 2~4×。
│   - 层间共享缓存：部分层可共享 K/V 投影或直接复用
│     下层缓存，进一步压缩总量（实验性）。
│
└── 最终闭环：架构压缩 + 分页管理 + 量化，使得推理系统
    可在有限显存下支撑极大规模批次与长上下文。

结论
       KV 缓存以存储换计算，是 Transformer 解码的标准加速手段。
       其演化路线从简单缓存 → 减少单条缓存体积 (MQA/GQA) →
       消灭分配碎片 (PagedAttention) → 量化压缩，始终环绕
       显存容量与带宽这一核心约束递进。

参见
       flash-attention(7), pageattention(7), vllm-arch(7)

版本
       基础概念，随 Transformer 架构与推理系统共同演进。

KV-CACHE(7)                                                2026-05-02
```