# TRANSFORMER(7)

## 名称

transformer - 基于自注意力的序列到序列神经网络架构

## 概述

Transformer 摒弃循环结构，完全依赖注意力机制建模序列内元素间的全局依赖，  
实现并行计算与长程交互。以下按“问题→解决→新问题”链条逐层推导其核心原理。

## 详解

### 1. 问题：序列建模中，如何高效捕获远距离依赖？

RNN 受限于时序递推，长距离梯度消失，且无法并行。  
解决思路：放弃状态传递，让序列中任意两个位置直接交互。  
→ **自注意力 (Self-Attention)**

- 对输入序列 X ∈ ℝ^{n×d} (n 个 token，维度 d)，  
通过可学习矩阵 W_Q, W_K, W_V 投影到三个空间：  
Q = X W_Q    (查询, Query)  
K = X W_K    (键, Key)  
V = X W_V    (值, Value)  
Q, K, V ∈ ℝ^{n×d_k} (d_k 为投影维度)
- 注意力权重由 Q 与 K 的相似度决定，经 Softmax 归一化后加权 V：  
Attention(Q, K, V) = softmax( Q K^T / √d_k ) V  
缩放因子 √d_k 防止点积过大导致 Softmax 梯度消失。

→ 输出中的每个位置都融合了全局信息，计算可完全并行。

### 2. 问题：自注意力对位置不敏感，如何注入序列顺序信息？

“A 依赖 B” 与 “B 依赖 A” 在无位置信息时不可区分。  
→ **位置编码 (Positional Encoding)**

- 为正弦/余弦固定编码，或可学习的 Embedding：  
PE(pos, 2i)   = sin(pos / 10000^{2i/d})  
PE(pos, 2i+1) = cos(pos / 10000^{2i/d})  
其中 pos 为位置，i 为维度索引。
- 将位置编码直接加到输入嵌入上：  
X' = Embedding(token) + PE(pos)  
使模型能感知 token 的绝对与相对位置。

### 3. 问题：单套注意力可能只捕获一种关系模式，如何丰富表达？

单头注意力类似于仅用一种特征提取器，表达力受限。  
→ **多头注意力 (Multi-Head Attention)**

- 并行执行 h 个独立的注意力头，每个头使用不同的投影矩阵：  
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)  
W_i^Q, W_i^K ∈ ℝ^{d×d_k}, W_i^V ∈ ℝ^{d×d_v}，通常 d_k = d_v = d/h
- 拼接所有头的结果并再次线性投影：  
MultiHead(Q, K, V) = Concat(head_1, …, head_h) W_O  
W_O ∈ ℝ^{h·d_v × d}

→ 不同头可关注不同子空间（如语法依赖、语义相关、位置距离等）。

### 4. 问题：纯注意力仅是线性加权，如何引入非线性和 token 自身变换？

每个位置的输出经注意力汇聚后，仍需独立进行特征转换。  
→ **逐位置前馈网络 (Position-wise Feed-Forward Network, FFN)**

- 对每个位置施加相同的两层全连接网络，中间含 ReLU 激活：  
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2  
W_1 ∈ ℝ^{d×d_ff}, W_2 ∈ ℝ^{d_ff×d}，通常 d_ff > d (如 4d)

→ 提供非线性变换，扩展模型容量。

### 5. 问题：深层网络训练中，梯度易消失/爆炸，信号难以直达底层？

多层堆叠时，数值波动和梯度衰减严重，收敛困难。  
→ **残差连接 (Residual Connection) 与 层归一化 (Layer Normalization)**

- 每个子层（注意力或 FFN）输出与原输入相加后再归一化：  
LayerNorm( x + Sublayer(x) )  
其中 Sublayer 为多头注意力或前馈网络。
- 层归一化在特征维度上计算均值与方差：  
LayerNorm(x) = γ ⊙ (x - μ)/√(σ² + ε) + β  
μ, σ² 为特征维度的统计量，γ, β 为可学习缩放与偏置。

→ 保证训练稳定性，加速收敛。

### 6. 问题：解码时如何避免看到未来信息，同时建立编码-解码交互？

自回归生成中，解码器必须防止位置 i 关注 j > i。  
→ **因果掩码 (Causal Mask) 与 交叉注意力 (Cross-Attention)**

- 在解码器自注意力中，给注意力分数矩阵加上掩码 M，使其上三角为 -∞：  
Attention(Q, K, V) = softmax( QK^T/√d_k + M ) V  
M[i, j] = 0  if j ≤ i, else -∞  
使 Softmax 后未来位置的权重为 0。
- 解码器额外插入交叉注意力层，Q 来自解码器，K, V 来自编码器输出：  
CrossAttention(Q_dec, K_enc, V_enc)  
使解码器能有选择地关注输入序列的相关部分。

### 7. 整体架构

Transformer 由编码器与解码器堆叠而成：

编码器 (N 层)：  
Input → Embedding + Positional Encoding  
for i = 1..N:  
x = LayerNorm( x + MultiHead_SelfAttention(x) )  
x = LayerNorm( x + FFN(x) )  
→ Encoder Output

解码器 (N 层)：  
Input → Embedding + Positional Encoding  
for i = 1..N:  
y = LayerNorm( y + Masked_MultiHead_SelfAttention(y) )  
y = LayerNorm( y + MultiHead_CrossAttention(y, Encoder_Output) )  
y = LayerNorm( y + FFN(y) )  
→ y → Linear + Softmax → Output Probabilities

其中 d_model 为统一维度，所有子层及嵌入层均遵循该维度。

## 参见

- "Attention Is All You Need" (Vaswani et al., 2017)
- BERT, GPT 等衍生模型