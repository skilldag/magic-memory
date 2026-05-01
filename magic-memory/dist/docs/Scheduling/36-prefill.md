# VLLM-PREFILL(7) — LLM Reasoning Engine Manual

## NAME

`vllm-prefill` — 大语言模型推理的预填充阶段：KV 缓存的构造与优化

## SYNOPSIS

```text
LLMEngine | --[prompt]--> Prefill --> KV_Cache --> Decode --> tokens
              O(N²)        O(N²)       O(N)        O(N)
```

## DESCRIPTION

`vllm-prefill` 是 LLM 自回归推理的第一阶段。系统接收 Prompt Token 序列

```text
X = [x_0, x_1, ..., x_{L-1}],   dim(X) = [L, d]
```

并一次性完成 L 个位置的 Attention 前向计算，生成完整的 Key-Value 缓存。

```text
Q, K, V = X·W_Q, X·W_K, X·W_V           # Linear projections
S = Q·K^T / √d_k                          # Score matrix, shape [L, L]
A = softmax(S, dim=-1, causal_mask=True)   # Lower-triangular causal mask
O = A·V                                    # dim [L, d]
```

此后每个 Transformer 层的 K, V 被持久化至显存，构成 **KV Cache**，供后续 Decode 阶段以 O(L) 复杂度逐 token 自回归生成。

### 核心矛盾

Prefill 的计算复杂度为 **O(L²·d)**，Decode 仅为 **O(L·d)**。随着 Prompt 长度 L↑，Prefill 的时延占比急剧上升，成为首 Token 时延（TTFT）的主要瓶颈。-

```text
TTFT = T_prefill(L) ≈ α·L²·d / Compute  ,  α: const
```

**问题一**：长 Prompt 场景下 Prefill 计算量过大，TTFT 不可接受，如何削减？

---

## CHUNKED PREFILL — 分块预填充

### 概念

将长度为 L 的 Prompt 切分为 K 个 Chunk：

```text
L = Σ_{i=1}^{K} l_i,   l_i ≤ max_tokens_per_chunk
```

每个 Chunk 独立进行 Prefill 前向计算，中间 KV 累积，仅最后一个 Chunk 完成后产生首 Token。[-9](https://docs.vllm.ai/en/v0.4.2/models/performance.html)[-11](https://support.huaweicloud.com/intl/zh-cn/bestpractice-modelarts/modelarts_llm_infer_5906026.html)

### 优势

- **TTFT 感知提前**：用户无需等待全部 L 个 token 计算完毕即可感知进度
- **Compute-Memory 混合调度**：Prefill Chunk（计算密集）与 Decode（访存密集）可混入同一 Batch，GPU 利用率 ↑[-9](https://docs.vllm.ai/en/v0.4.2/models/performance.html)
- **ITL 稳定性**：避免单次长 Prefill 独占 GPU 时间片导致 Decode 饥饿[-9](https://docs.vllm.ai/en/v0.4.2/models/performance.html)

### 调度策略

```text
┌─ Scheduler ─────────────────────────────────────┐
│  token_budget = max_num_batched_tokens           │
│                                                  │
│  WHILE pending != ∅:                             │
│    batch ← pending_decode                        │
│    remaining = token_budget - Σ|batch|           │
│    WHILE remaining > 0 AND pending_prefill ≠ ∅:  │
│      p ← pending_prefill.pop()                   │
│      IF |p| ≤ remaining:                         │
│        batch ← batch ∪ {p}                       │
│      ELSE:                                       │
│        p_chunk = p[:remaining]                   │
│        batch ← batch ∪ {p_chunk}                 │
│        pending_prefill.push(p[remaining:])       │
│    forward(batch)                                │
│    last_chunk? → sample_first_token              │
└──────────────────────────────────────────────────┘
```

- `max_num_batched_tokens`：每 Step 最大 Token 预算，典型值 512（ITL 最优）或 >2048（吞吐量最优）[-9](https://docs.vllm.ai/en/v0.4.2/models/performance.html)
- 优先调度 Decode 请求，剩余 Token 配额分配给 Prefill Chunk[-9](https://docs.vllm.ai/en/v0.4.2/models/performance.html)
- 约束：该特性不能与 Prefix Cache 同时启用[-11](https://support.huaweicloud.com/intl/zh-cn/bestpractice-modelarts/modelarts_llm_infer_5906026.html)

### 计算一致性

尽管 Attention 复杂度随 Chunk 中 token 位置增长（需关注的前缀越来越长），vLLM 的 PagedAttention 内核通过硬件级优化（kernel fusion、shared memory tiling）使得各 Chunk 的 wall-clock 耗时近乎恒定。-

**问题二**：Chunked Prefill 缓解了长 Prompt 的 TTFT 问题，但系统 Prompt / 多轮对话场景下，不同请求共享大量相同的 Prompt 前缀，重复计算浪费算力，如何消除？

---

## PREFIX CACHING — 自动前缀缓存

### 概念

vLLM 采用 **Automatic Prefix Caching (APC)**：缓存已处理请求的 KV Cache Block，当新请求的前缀与历史请求匹配时直接复用，无需重新计算。[-13](https://docs.vllm.com.cn/en/latest/design/prefix_caching/#__codelineno-0-1)

```text
Request A: P_sys + P_user_A   →  compute KV(P_sys), KV(P_user_A)
Request B: P_sys + P_user_B   →  reuse KV(P_sys), compute KV(P_user_B) only
Saving  = |P_sys| · (2·n_layers·n_heads·d_head) · dtype_bytes
```

### 块粒度哈希

vLLM 以 **Block**（默认 16 tokens / Block）为缓存单位，仅缓存完整 Block。[-23](https://developer.aliyun.com/article/1680330)[-13](https://docs.vllm.com.cn/en/latest/design/prefix_caching/#__codelineno-0-1)

```text
Hash(Block_i) = SHA256(
    ParentHash(Block_{i-1}),    # 链式依赖确保前缀唯一性
    Tokens(Block_i),            # 块内 token 元组，降低碰撞
    ExtraHashes                 # LoRA ID / 多模态哈希 / cache_salt
)
```

- 只有父块哈希匹配成功才会继续匹配子块，形成 **Radix Tree** 结构[-28](https://cloud.tencent.cn/developer/article/2424704?policyId=1004)
- `cache_salt` 提供多租户隔离：不同租户的相同前缀生成不同哈希，防止时序侧信道攻击[-13](https://docs.vllm.com.cn/en/latest/design/prefix_caching/#__codelineno-0-1)
- v0.11 起默认使用 SHA256，碰撞概率可忽略；也支持 xxHash 以换取更高性能[-13](https://docs.vllm.com.cn/en/latest/design/prefix_caching/#__codelineno-0-1)

### 命中判定

```text
Input:  prompt_tokens[0:N]

matched_blocks = 0
FOR i = 0 TO floor(N / BLOCK_SIZE) - 1:
    block_tokens = prompt_tokens[i·B : (i+1)·B]
    h = SHA256(parent_hash, block_tokens, extra)
    IF h ∈ CacheStore:
        reuse_KV(block_id = i)
        matched_blocks++
    ELSE:
        BREAK  # 一旦不匹配，后续块皆需重新计算

computed_tokens = N - matched_blocks * BLOCK_SIZE
```

### 多模态扩展

图像等非文本输入经 processor 生成图像哈希，嵌入到 Block Hash 的 ExtraHashes 字段，确保不同图像（即使 placeholder 相同）生成独立缓存。[-13](https://docs.vllm.com.cn/en/latest/design/prefix_caching/#__codelineno-0-1)

**问题三**：Prefix Caching 在大规模部署时，GPU 显存有限，缓存淘汰策略如何保证命中率？更根本的——Prefill 和 Decode 对硬件资源的需求截然不同（计算密集 vs 访存密集），能否从架构层面解耦？

---

## PD DISAGGREGATION — 预填充/解码分离

### 动机

Prefill 与 Decode 的硬件瓶颈正交：

```text
Prefill:  compute-bound  →  需要高 FLOPS（Tensor Core 密集）
Decode:   memory-bound   →  需要高显存带宽（HBM 密集）
```

单实例混合调度导致：

- Decode 过程中被 Prefill 插入，ITL 尾部（Tail ITL）显著升高[-26](https://discuss.vllm.ai/t/disaggregated-prefilling-tail-itl/2386)[-2](https://docs.vllm.com.cn/en/latest/features/disagg_prefill/#__codelineno-0-1)
- Prefill 期间 Decode 请求排队，TTFT 恶化
- 无法为 P/D 分别调优并行策略（TP/PP/EP）[-2](https://docs.vllm.com.cn/en/latest/features/disagg_prefill/#__codelineno-0-1)

### 架构

```text
               ┌─────────────────┐
User Request →│  Global Proxy    │
               └───┬─────────┬───┘
                   │         │
            ┌──────▼──┐ ┌───▼──────┐
            │ P Node  │ │ D Node   │
            │ (Prefill)│ │ (Decode) │
            └────┬─────┘ └───┬──────┘
                 │  KV Cache │
                 └───P2P─────┘
```

- **P Node**：接收完整 Prompt，执行 Prefill 生成 KV Cache，不产生输出 Token
- **KV Transfer**：通过 P2P（PyNcclConnector / MooncakeConnector / NixlConnector）将 KV Cache 从 P 传输至 D[-4](https://docs.vllm.ai/projects/ascend/zh-cn/main/developer_guide/Design_Documents/disaggregated_prefill.html)[-5](https://docs.vllm.ai/projects/ascend/zh-cn/v0.11.0-dev/developer_guide/feature_guide/disaggregated_prefill.html)
- **D Node**：等待远程 KV Cache 就绪后执行纯 Decode，返回生成 Token

### 连接器语义

- `MooncakeConnector`（拉取模式）：D 节点主动从 P 节点拉取 KV；P 完成后延迟释放等待拉取[-4](https://docs.vllm.ai/projects/ascend/zh-cn/main/developer_guide/Design_Documents/disaggregated_prefill.html)
- `MooncakeLayerwiseConnector`（推送模式）：P 节点按层推送到 D 节点，降低 D 端等待延迟[-4](https://docs.vllm.ai/projects/ascend/zh-cn/main/developer_guide/Design_Documents/disaggregated_prefill.html)
- `NixlConnector`：全异步发送/接收，适用大规模异步调度场景[-2](https://docs.vllm.com.cn/en/latest/features/disagg_prefill/#__codelineno-0-1)

### 权衡

- ✅ **TTFT/ITL 独立调优**：分别调整 P/D 的 TP size、实例数量[-2](https://docs.vllm.com.cn/en/latest/features/disagg_prefill/#__codelineno-0-1)
- ✅ **Tail ITL 可控**：Decode 不再被 Prefill 中断[-2](https://docs.vllm.com.cn/en/latest/features/disagg_prefill/#__codelineno-0-1)
- ✅ **资源利用率**：P 和 D 可部署在不同硬件配置上
- ❌ **不提升吞吐量**：总体计算量不变，且引入 KV 传输开销[-2](https://docs.vllm.com.cn/en/latest/features/disagg_prefill/#__codelineno-0-1)
- ❌ **系统复杂度**：额外引入全局代理、连接器管理、故障恢复

**问题四**：架构层面解耦了 P/D，但 P 节点本身处理海量并发请求时仍面临 KV Cache 显存膨胀——能否在块级别做到请求间零拷贝共享？

---

## 端到端数据流总览

```text
                   [PageAttention Block Manager]
                            │
Tokenizer ──► Prefill ──────┤
                │            ├─► KV Blocks (hashed, indexed)
                │            │       │
                ▼            │       ▼
             [Prefix Cache Hit?]   [Block Table per Request]
                │                          │
      ┌────YES──┴──NO───┐                  │
      ▼                 ▼                  ▼
 Reuse KV          Compute KV ────►   Decode Step
      │                 │                  │
      └────────┬────────┘                  ▼
               └────────────► Sample ──► Token
```

### 关键数据结构

```text
KVBlock:
    block_id:      uint32
    token_range:   [start, end)
    parent_hash:   SHA256
    ref_count:     atomic<uint32>
    data:          Tensor[block_size, num_kv_heads, head_size]

BlockTable:
    request_id → [block_id_0, block_id_1, ..., block_id_m]

CacheStore:
    hash → block_id   (HashMap with LRU eviction on OOM)
```

### 性能指标公式

```text
TTFT = T_prefill(computed_tokens) + T_kv_lookup(prefix_hits)
     ≈ β·(N - H·B)²·d / FLOPS_prefill

ITL  = T_decode_step ≈ γ·(H·B + current_len)·d / BW_mem

where:  N = total prompt tokens
        H = number of prefix cache hits (blocks)
        B = block size (default 16)
        β, γ: system-dependent constants
```

## COMPATIBILITY MATRIX

| Feature | Chunked Prefill | Prefix Cache | PD Disaggregation | Notes |
| --- | --- | --- | --- | --- |
| Chunked Prefill | — | ✗ | ✓ | APC + CP 不能同时启用 |
| Prefix Cache | ✗ | — | ✓ | PD 分离下 P 和 D 均可复用 |
| PD Disaggregation | ✓ | ✓ | — | KV 通过 P2P 传输 |

## SEE ALSO

`vllm-decode`(7), `vllm-scheduler`(7), `paged-attention`(7), `kv-cache`(7)

*ArXiv:* [2401.08671] Splitwise — Efficient LLM Inference with P/D Separation, [2312.07104] SGLang — RadixAttention for Prefix-Aware Scheduling

*vLLM Docs:* `docs.vllm.ai/en/latest/features/disagg_prefill.html`, `docs.vllm.ai/en/latest/design/prefix_caching.html`

## COLOPHON

This page is part of the `vllm-docs` project.  Report issues to `https://github.com/vllm-project/vllm/issues`.