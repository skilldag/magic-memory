31-DRAFT-TOKEN(7) — Speculative Decoding Core Mechanism
NAME
31-draft-token — 投机解码中草稿令牌的批量生成、并行验证与接受-拒绝采样机制。

DESCRIPTION
大语言模型的逐令牌自回归解码面临硬件并行算力利用不足的瓶颈：每个 forward pass 仅产出 1 个 token，GPU 计算单元大量闲置。31-draft-token 范式以空间换时间，通过一次性生成最多 31 个候选 token 并交由目标模型并行验证，将单步产出 token 数的数学期望提升至 ⌊ακ⌋ 以上，其中 α 为接受率，κ 为草稿步数。

DETAILS
Q0: 自回归解码为何低效？
text
Problem:
    Autoregressive decoding of LLM with K_h hidden_dim, L layers:
        T_per_token ∝ L · K_h²                                (1)
    即每个 token 的延迟与模型参数量成线性关系。
    每次 forward pass 仅产出 1 个 token，GPU 利用率 < 10%。
Q1: 如何突破逐令牌串行瓶颈？
text
Solution: Speculative Decoding — Draft-then-Verify 范式。

Sub-concept: Draft Model M_d
    · 参数量 P_d << P_t (target model)，推理速度 v_d >> v_t
    · 在给定前缀 x_<t> 条件下自回归生成 K 个候选 token:
        x_draft = [x_t, x_{t+1}, ..., x_{t+K-1}]             (2)
    · K 的典型取值: 3–8 (默认), 31 (上限), 取决于 GPU 显存与接受率。

Sub-concept: Target Model M_t 并行验证
    · 单次 forward pass 输入 [x_<t>; x_draft]，产出 K+1 步的概率分布:
        ∀k ∈ [0, K]: P_t(· | x_<t>, x_draft[:k])              (3)
    · 并行性: 将 K+1 个位置沿 batch 维度展开，利用 GPU SIMD 并行。
    · 延迟: T_verify ≈ T_per_token · (1 + ε)，其中 ε << 1。
Q2: 验证后的草稿 token 如何裁定接受或拒绝？
text
Problem:
    草稿模型分布 q(x) 与目标模型分布 p(x) 存在偏差。
    直接拼接可能导致输出分布偏移，破坏生成质量。
    必须设计无偏接受机制，保证最终输出分布 = p(x)。

Solution: Rejection Sampling (Leviathan et al., 2023)。

Sub-concept: Per-token Acceptance Criterion
    对于位置 i ∈ [0, K-1]，生成随机数 r ~ Uniform(0, 1):

        if   r < min(1, p_i(x) / q_i(x)): accept x_i          (4)
        else: reject x_i and all subsequent tokens

    · 若 q_i > p_i，按比例接受以保证无偏:
        P(accept | x_i) = min(1, p_i(x) / q_i(x))             (5)
    · 一旦某个 token 被拒绝，从修正分布采样:
        x_repl ~ norm(max(0, p_i - q_i))                      (6)
    · 拒绝后所有后续草稿 token 被丢弃 (discard)。

Sub-concept: Acceptance Rate α
    · 定义: α = E[accept_i] = Σ min(p(x), q(x))              (7)
    · 经验值: α ∈ [0.7, 0.95]，取决于领域对齐程度。
    · 每步产出 token 数期望: E[#tokens] = (1-α^{K+1})/(1-α)  (8)
    · 理论加速比: speedup = E[#tokens] · T_per_token / T_verify
Q3: 参数 K=31 的工程动机与约束是什么？
text
Problem:
    更大的 K 提升峰值吞吐，但引入边际效用递减:
        · 后续草稿 token 的接受率为累积乘积，递远衰减:
            P(accept all K) = Π_j=0^{K-1} α_j ≈ α^K          (9)
        · α^31 ≈ 0.04 (α=0.9 时)，大量算力被浪费。

Solution: 31-draft-token 作为硬件友好上限。

Sub-concept: GPU 显存约束
    · KV-cache 扩容: 每增加一个草稿 token，KV-cache 扩展 K_h · L 维度。
    · 31 tokens → KV-cache 增量 ≈ 31 · K_h · L · 2 (bytes, FP16)
    · A100-80GB: 31 tokens @ Llama-70B ≈ 额外占用 ~1.2 GB。

Sub-concept: 流水线优化
    · Draft 阶段: M_d 自回归生成 31 tokens (低延迟)。
    · Verify 阶段: M_t 并行验证 32 位置 (含前缀)，1 次 forward pass。
    · 延迟模型:
        T_step = T_draft(31) + T_verify(32)
               ≈ 31/v_d + T_per_token                         (10)
    · 加速比上界:
        speedup_max = K+1 = 32 (理论极限，α=1 时)            (11)
    · 实际加速比:
        speedup_eff = E[#tokens] / (1 + 31 · (v_t / v_d))   (12)
Q4: 如何进一步提升草稿质量 (提高 α)？
text
Problem:
    独立草稿模型分布 q 与目标 p 偏差导致低接受率。
    α ↓ → E[#tokens] ↓ → 加速比劣化。

Solution: Tree-structured Drafting。

Sub-concept: Draft Token Tree
    · 每步不再生成线性链，而是生成树宽 B、深度 D 的 tree:
        total_nodes = (B^{D+1} - 1) / (B - 1) ≤ 31           (13)
    · 典型配置: B=2, D=4 → 31 nodes; B=3, D=3 → 40 nodes。

Sub-concept: Tree Verification
    · M_t 对树的所有路径并行评分。
    · 选择累积概率最高的路径作为输出。
    · α_tree > α_chain，因多条路径提供备选方案。
    · 在相同 31-token 预算下，树结构比线性链更鲁棒。
Q5: 31-draft-token 流水线的全局状态机？
text
State Transition:

    [Prefix Accumulation]  →  [Draft Generation]
           ↑                        ↓
           |                  [Tree/Chain Drafting: ≤31 tokens]
           |                        ↓
           |                  [Target Model Verification]
           |                        ↓
           |                  [Rejection Sampling: accept/reject]
           |                        ↓
           +—————— [Append accepted tokens to output]
           |                        ↓
           +—————— [Discard rejected & subsequent tokens]
           |                        ↓
           +—————— [Resample from corrected distribution]
                                    ↓
                            [Continue or EOS]

Pipeline Formula:
    Output = Σ_{step=1}^{S} Accept( Draft( Prefix_{step}, M_d, K ), M_t, p )
    where Accept() implements Eq.(4)–(6).

Invariant:
    P(output ∈ A) = P_{M_t}(A)  ∀ measurable set A.
    ⇒ 输出分布与 M_t 原生分布无偏，加速同时无损质量。
IMPLEMENTATION NOTES
Draft Model 选择: 建议 P_d / P_t ∈ [0.02, 0.15]，同一词表 (tokenizer 对齐)。

K 值调优: 在开发集上扫描 K ∈ {3, 5, 7, 11, 15, 23, 31}，选取吞吐 (tokens/s) 峰值。

显存管理: 树状草稿需 pre-allocate KV-cache，建议使用 paged attention 技术动态分配。

拒绝采样实现: 使用 Gumbel-Max 技巧或直接蒙特卡洛采样；避免数值溢出 (log-prob 空间计算)。

终止条件: EOS token 在草稿中出现时立即截断，不再生成后续位置。

SEE ALSO
speculative-decoding(7), rejection-sampling(3), kv-cache(5), llama.cpp(1), tensorrt-llm(1)

