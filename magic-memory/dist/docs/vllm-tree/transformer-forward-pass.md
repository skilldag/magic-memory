# Transformer 完整前向推导知识树

> 从输入文本到下一个 token 概率的逐行推导

---

## 来时路

```
计算环节 → Transformer → 前向传播 → 完整推导链
```

从 Transformer 的前向传播过程出发，推导从输入到输出的完整计算链。

---

## 根问题：给定输入文本，如何逐行算出下一个 token 的概率？

### 第 0 步：数据准备

#### Q: 输入是什么形状？
**A**: `token_ids: [Batch, Seq_Len]` 例如 `[1, 4]` 代表 "我 爱 你 <eos>`

#### Q: 目标 labels 怎么构造？
**A**: 
- `labels = token_ids[:, 1:]` 即 "爱 你 <eos>"
- 模型输入 = `token_ids[:, :-1]` 即 "我 爱 你"
- 这是标准的 teacher forcing 偏移

#### Q: 变长句子如何组成 Batch？
**A**: 同一个 Batch 内的句子必须对齐到相同长度，但不同 Batch 之间可以长度不同。

**为什么需要 Batch？**
- 技术上单个句子可以驱动一次前向+反向+更新
- 但单句梯度方差巨大，训练极不稳定
- Batch 的核心作用：用平均梯度平滑更新方向，稳定收敛

**为什么同 Batch 必须等长？**
- 输入 GPU 的张量必须是矩形的
- `input_ids: [Batch, Seq_Len]`
- 如果句子长度不一，无法堆叠成二维矩阵

**怎么拉齐？Padding**
- 确定 Batch 内最大长度 `L_max`
- 每个句子尾部填充特殊 token `[PAD]` 至 `L_max`
- 结果：所有句子长度 = `L_max`

**例子**：
```
句子1: [15, 208, 99, PAD, PAD]
句子2: [7,  33,  12,  45,  88 ]
句子3: [9,  10,  11,  12,  PAD]
Batch 形状 → [3, 5]
```

**怎么屏蔽 PAD？**
- 构造 Attention Mask：`mask ∈ {0, 1}^{Batch × Seq_Len}`
- `1 = 真实 token`，`0 = PAD`
- 在注意力计算时：`Scores = Scores.masked_fill(mask == 0, -1e9)`
- 在计算 Loss 时：`loss = CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)`

#### Q: 超参数定义？
```python
d_model = 768      # 隐藏维度
num_heads = 12     # 注意力头数
d_head = 64        # 每头维度 (768/12)
num_layers = 12    # Transformer 层数
vocab_size = 10000 # 词表大小
max_len = 2048     # 最大序列长度
```

---

### 第 1 步：Embedding 查表 + 位置编码

#### Q1.1: Token Embedding 怎么查？
**A**: 
- `W_e ∈ R^{vocab_size × d_model}` (可学习参数)
- `X = W_e[input_ids]` 形状 `[B, T, d_model]`
- 例如输入 `[15, 208, 99]` → `[1, 3, 768]`

**符号解释**：
- `∈` 是"属于"符号
- `R` 代表实数集（所有小数、整数、有理数、无理数……）
- `W_e ∈ R^{vocab_size × d_model}` 意思是：W_e 是一个 vocab_size 行、d_model 列的矩阵，里面每个元素都是实数

#### Q1.2: 位置编码怎么加？（以 RoPE 为例说明两种方式）

**如果用可学习绝对位置**：
- `W_pos ∈ R^{max_len × d_model}` (可学习参数)
- `pos_ids = [0, 1, 2]`
- `X = X + W_pos[pos_ids]` 形状 `[1, 3, 768]`

**如果用 RoPE（旋转位置编码）**：
- 这里不在输入层加位置，而是在每一层的注意力中对 Q 和 K 做旋转操作
- 输入层 X 保持纯语义
- （详见注意力子层的 RoPE 推导）

#### Q1.3: Dropout？
**A**: `X = Dropout(X, p=0.1)` 仅训练时，推理时跳过

---

### 第 2 步：循环进入 N 层 Transformer Block

```
For layer_id in 1..num_layers:
```

#### [子层 2.1: 带残差的注意力]

##### Q2.1.1: LayerNorm 在哪？(Pre-LN 现代标准)
**A**: `X_norm = LayerNorm(X)` 形状 `[B, T, d_model]`

##### Q2.1.2: 怎么生成多头 Q, K, V？
**A**: 
- `W_q, W_k, W_v ∈ R^{d_model × d_model}` (每层独立参数)
- `Q = X_norm @ W_q` 形状 `[B, T, d_model]`
- `K = X_norm @ W_k` 形状 `[B, T, d_model]`
- `V = X_norm @ W_v` 形状 `[B, T, d_model]`

##### Q2.1.3: 如何切成多头？
**A**: 
```
Q = Q.view(B, T, num_heads, d_head).transpose(1, 2)
→ [B, num_heads, T, d_head]
K = K.view(B, T, num_heads, d_head).transpose(1, 2)
→ [B, num_heads, T, d_head]
V = V.view(B, T, num_heads, d_head).transpose(1, 2)
→ [B, num_heads, T, d_head]
```

##### Q2.1.4: RoPE 旋转位置编码怎么作用在 Q 和 K 上？

**预计算频率**：
- `θ_i = 10000^(-2i/d_head)`, `i=0,1,...,d_head/2-1`
- 每个 `θ_i` 用于 `d_head` 中的第 `(2i, 2i+1)` 维

**对每个位置 m，计算旋转角**：`m × θ_i`
- 位置索引 `m = [0, 1, 2, ...]`

**对 Q 和 K 的每一对维度 (2i, 2i+1) 施加旋转**：
```
[x_2i]   →   [cos(mθ_i)  -sin(mθ_i)] [x_2i]
[x_2i+1]       [sin(mθ_i)   cos(mθ_i)] [x_2i+1]
```
- 这是按位置 m 对 Q 的行旋转，对 K 的行也相同
- 形状不变：`[B, H, T, d_head]`

##### Q2.1.5: 注意力分数怎么算？
**A**: 
```
Scores = Q @ K^T / sqrt(d_head)
→ [B, H, T, T]  例如 [1, 12, 3, 3]
Scores_ij = 第 i 个 query 对第 j 个 key 的分数
```

##### Q2.1.6: Causal Mask 怎么加？
**A**: 
```
mask = torch.triu(ones(T, T), diagonal=1) * (-1e9)
mask = [[0,    -inf, -inf],
        [0,    0,    -inf],
        [0,    0,    0   ]]
Scores = Scores + mask  广播到 [B, H, T, T]
效果：query i 只能看到 key j ≤ i
```

##### Q2.1.7: Softmax 归一化
**A**: 
```
Attn = softmax(Scores, dim=-1)  形状 [B, H, T, T]
Attn[i,j] = 第 i 个 token 对第 j 个 token 的关注权重
且对于所有 j > i，Attn[i,j] = 0
```

##### Q2.1.8: 用注意力权重对 V 加权求和
**A**: 
```
Context = Attn @ V   形状 [B, H, T, d_head]
含义：每个头独立地从 V 中提取信息
```

##### Q2.1.9: 多头如何合并？
**A**: 
```
Context = Context.transpose(1, 2).reshape(B, T, d_model)
→ [B, T, d_model]
物理含义：把 12 个头各自的 64 维拼接回 768 维
```

##### Q2.1.10: 输出投影
**A**: 
```
W_o ∈ R^{d_model × d_model}
Attn_Out = Context @ W_o   形状 [B, T, d_model]
```

##### Q2.1.11: 第一个残差连接
**A**: 
```
Attn_Out = Dropout(Attn_Out, p=0.1)
X = X + Attn_Out   形状 [B, T, d_model]
注意：这里的 X 是进入子层之前的原始 X，
不是经过 LayerNorm 的 X_norm。这是 Pre-LN 的要点。
```

#### [子层 2.2: 带残差的 FFN]

##### Q2.2.1: LayerNorm (Pre-LN)
**A**: `X_norm = LayerNorm(X)` 形状 `[B, T, d_model]`

##### Q2.2.2: FFN 第一层 (升维)
**A**: 
```
W1 ∈ R^{d_model × d_ff}, b1 ∈ R^{d_ff}
通常 d_ff = 4 * d_model = 3072
H = X_norm @ W1 + b1    形状 [B, T, d_ff]
```

##### Q2.2.3: 非线性激活
**A**: 
```
H_act = GELU(H)  形状 [B, T, d_ff]
GELU(x) = x * Φ(x) ≈ 0.5x * (1 + tanh(√(2/π)*(x+0.044715x³)))
近似效果：正数接近线性，负数接近零（平滑版ReLU）
```

##### Q2.2.4: FFN 第二层 (降维)
**A**: 
```
W2 ∈ R^{d_ff × d_model}, b2 ∈ R^{d_model}
FFN_Out = H_act @ W2 + b2   形状 [B, T, d_model]
```

##### Q2.2.5: 第二个残差连接
**A**: 
```
FFN_Out = Dropout(FFN_Out, p=0.1)
X = X + FFN_Out   形状 [B, T, d_model]
再次：加的是进入子层之前的 X，不是 X_norm
```

---

### 第 3 步：最终输出层

#### Q3.1: 最后的 LayerNorm
**A**: 
```
X = LayerNorm(X)    形状 [B, T, d_model]
注意：原始 GPT-2/LLaMA 在最后一层之后有额外 LayerNorm
```

#### Q3.2: 投影到词表维度（LM Head）
**A**: 
```
W_lm ∈ R^{vocab_size × d_model}  通常与 W_e 共享权重
logits = X @ W_lm^T     形状 [B, T, vocab_size]
含义：每个位置对下一个 token 的未归一化概率
```

#### Q3.3: 计算损失（仅训练时）
**A**: 
```
logits = logits[:, :-1, :]  形状 [B, T-1, vocab_size]
取前 T-1 个位置的预测（最后一个位置无下一个token）

labels = input_ids[:, 1:]  形状 [B, T-1]
后 T-1 个 token 作为答案

loss = CrossEntropy(logits.reshape(-1, vocab_size), labels.reshape(-1))
标量，所有位置的平均交叉熵
```

---

## 关键形状变化一览表

| 步骤 | 操作 | 输入形状 | 输出形状 |
|------|------|----------|----------|
| Token Embed | 查表 | [B, T] | [B, T, 768] |
| 线性投影 Q/K/V | X@W | [B, T, 768] | [B, T, 768] |
| 多头切分 | .view+transpose | [B, T, 768] | [B, 12, T, 64] |
| RoPE | 逐对旋转 | [B, 12, T, 64] | [B, 12, T, 64] |
| 注意力分数 | Q@K^T | [B, 12, T, 64] | [B, 12, T, T] |
| 加mask+Softmax | softmax(scores) | [B, 12, T, T] | [B, 12, T, T] |
| 加权求和 | Attn@V | [B, 12, T, T] | [B, 12, T, 64] |
| 多头合并 | .transpose+view | [B, 12, T, 64] | [B, T, 768] |
| FFN 升维 | X@W1 | [B, T, 768] | [B, T, 3072] |
| GELU | gelu(H) | [B, T, 3072] | [B, T, 3072] |
| FFN 降维 | H@W2 | [B, T, 3072] | [B, T, 768] |
| LM Head | X@W_lm^T | [B, T, 768] | [B, T, 10000] |

---

## 手写推导时的自查问题清单

当你合上笔记尝试手写时，用这些问题检验自己：

1. **形状方向**：每一步操作前后的形状我知道吗？
2. **残差原理**：为什么 `X = X + SubLayer(LayerNorm(X))`，加的是原始 X 不是 X_norm？
3. **Mask 细节**：Causal Mask 为什么是上三角 -inf？Softmax 后 j>i 的权重等于多少？
4. **多头本质**：`.transpose(1,2)` 是交换哪两个维度？为什么必须这么换才能并行？
5. **FFN 数值范围**：d_ff=3072，比 d_model=768 大四倍，为什么这很重要？
6. **激活函数选择**：GELU 在负数区域的输出是多少？接近 0 说明什么？
7. **输出投影**：从 [B, T, 768] 怎么变成 [B, T, 10000]？W_lm 的转置怎么回事？
8. **Loss 计算**：logits 为什么要切掉最后一个位置？labels 为什么要切掉第一个位置？

---

## 训练工程细节补充

### Batch 构造与 Padding

#### 为什么需要 Batch？
- 技术上单个句子可以驱动一次前向+反向+更新
- 但单句梯度方差巨大，训练极不稳定
- Batch 的核心作用：用平均梯度平滑更新方向，稳定收敛

#### 为什么同 Batch 必须等长？
- 输入 GPU 的张量必须是矩形的
- `input_ids: [Batch, Seq_Len]`
- 如果句子长度不一，无法堆叠成二维矩阵

#### 怎么用 Padding 拉齐？
- 确定 Batch 内最大长度 `L_max`
- 每个句子尾部填充特殊 token `[PAD]` 至 `L_max`
- 结果：所有句子长度 = `L_max`

#### 怎么屏蔽 PAD？
- 构造 Attention Mask：`mask ∈ {0, 1}^{Batch × Seq_Len}`
- `1 = 真实 token`，`0 = PAD`
- 在注意力计算时：`Scores = Scores.masked_fill(mask == 0, -1e9)`
- 在计算 Loss 时：`loss = CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)`

#### 不同 Batch 长度可以不同吗？
- 可以，每个 Batch 单独 Padding 到各自的 `L_max`
- GPU 只要求单次运算的张量等长，不要求跨批次等长

#### 长度差异大怎么办？
- **按长度分桶 (Bucketing)**：将语料按长度分组（如 0-128, 129-256, 257-512）
- **动态批处理 (Dynamic Batching)**：每个 epoch 先把数据按长度排序，再切 Batch
- **采样平衡**：控制长文本和短文本的比例

---

## 终极总结

### 完整前向传播流程

```
输入文本
    ↓
Tokenization → token_ids [B, T]
    ↓
Padding + Masking
    ↓
Embedding → X [B, T, d_model]
    ↓
For N 层 Transformer:
    ├─ LayerNorm
    ├─ Multi-Head Attention (with RoPE + Causal Mask)
    ├─ Residual Connection
    ├─ LayerNorm
    ├─ FFN (GELU)
    └─ Residual Connection
    ↓
Final LayerNorm
    ↓
LM Head → logits [B, T, vocab_size]
    ↓
CrossEntropy Loss (训练时)
```

### 关键洞察

1. **并行性**：Transformer 的革命性在于通过矩阵乘法实现完全并行，Causal Mask 保证因果性
2. **形状即一切**：每一步操作都严格遵循形状变化，形状错误是调试的第一信号
3. **残差连接**：Pre-LN 架构中，残差加的是原始 X，不是归一化后的 X_norm
4. **位置编码**：RoPE 通过旋转实现相对位置编码，天然支持外推
5. **工程细节**：Padding、Masking、Bucketing 是训练稳定性的关键

### 训练 vs 推理

| 维度 | 训练时 | 推理时 |
|------|--------|--------|
| 输入 | 完整序列 [B, T] | 逐步生成 |
| 并行 | 完全并行 | 串行（但可缓存 KV） |
| Loss | 计算交叉熵 | 不计算 |
| 梯度 | 反向传播 | 不需要 |
| 参数更新 | 梯度下降 | 不更新 |

这棵知识树可以作为手写推导的完整参考，涵盖了从数据准备到损失计算的所有关键步骤。
