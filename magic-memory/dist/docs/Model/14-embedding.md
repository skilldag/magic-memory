# EMBEDDING(7) — 核心原理的推导

## NAME

`embedding` — 从离散符号到连续语义向量的映射原理

## PROBLEM 0 · 符号的数值化鸿沟

- **问题**：语言、图谱、推荐等领域的原子实体 (词、节点、物品) 本质是离散符号，无法直接输入基于梯度优化的数值模型。
- **初解**：One‑Hot Encoding

  - 将符号 $w$ 表示为独热向量 $\mathbf{o} \in \{0,1\}^{|V|}$，其中 $|V|$ 为词汇量。
  - **致命缺陷**：

    - **维度灾难**：向量维数等于词汇量，稀疏且高维。
    - **语义零表达**：任意两个不同符号的独热向量正交，内积 $\mathbf{o}_i^\top \mathbf{o}_j = 0$，无法捕捉相似性。
- **新问题** → 如何构造低维、稠密、且能反映语义亲疏的向量？

---

## PROBLEM 1 · 低维语义空间的构建

- **核心概念**：**Embedding (嵌入)**

  - 定义一个查找映射 $f: V \to \mathbb{R}^d$，$d \ll |V|$。
  - 每个符号 $i$ 对应一个稠密向量 $\mathbf{v}_i \in \mathbb{R}^d$，称为嵌入向量。
  - 嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{|V| \times d}$ 可视作可训练的参数。
- **理论基础**：分布假说 (Distributional Hypothesis)

  - *“You shall know a word by the company it keeps.”*
  - **推论**：若两符号的上下文分布相似，则其嵌入向量应在几何上相近 (内积大、余弦相似度高)。
- **新问题** → 如何从原始语料或交互数据中，迫使嵌入矩阵 $\mathbf{E}$ 满足这一假说？

---

## PROBLEM 2 · 从共现中学习嵌入：预测任务

- **方法**：建立自监督预测目标，用上下文预测来注入分布信息。

  - **Skip‑gram (Word2Vec)**：给定中心词 $w_t$，预测局部窗口内的上下文词 $w_{t+j}$。
  - **最大对数似然**：

$$
\max \sum_{t} \sum_{-c \le j \le c, j \neq 0} \log P(w_{t+j} \mid w_t)
$$
  - **条件概率定义 (Softmax 单元)**：

$$
P(w_O \mid w_I) = \frac{\exp(\mathbf{v}'_{w_O} {}^\top \mathbf{v}_{w_I})}{\sum_{w \in V} \exp(\mathbf{v}'_w {}^\top \mathbf{v}_{w_I})}
$$

    - $\mathbf{v}_{w_I}$ — 输入嵌入 (中心词)，$\mathbf{v}'_{w_O}$ — 输出嵌入 (上下文词)。
- **瓶颈**：分母需对全词典 $V$ 求和，复杂度 $O(|V| \cdot d)$，词典庞大时不可行。
- **新问题** → 如何避免昂贵的全量归一化，高效近似训练？

---

## PROBLEM 3 · 高效似然近似：负采样

- **解决**：**Negative Sampling (NEG)**

  - 将多分类 (预测上下文词) 降级为二分类：区分“真实上下文对”与“随机噪声对”。
  - 对每个正样本 $(w_I, w_O)$，从噪声分布 $P_n(w)$ 采样 $k$ 个负样本 $w_i$。
  - **目标函数**：最大化

$$
\log \sigma(\mathbf{v}'_{w_O} {}^\top \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n} \left[ \log \sigma(-\mathbf{v}'_{w_i} {}^\top \mathbf{v}_{w_I}) \right]
$$

    - $\sigma(\cdot)$ — sigmoid 函数，$k$ 常取 5~20。
    - 计算复杂度降为 $O(k \cdot d)$，与 $|V|$ 无关。
- **成效**：得到高质量的**静态词嵌入** (Static Embeddings)。

  - 几何性质：线性语义关系，如 $\mathbf{v}_{king} - \mathbf{v}_{man} + \mathbf{v}_{woman} \approx \mathbf{v}_{queen}$。
- **新问题** → 静态嵌入为每一词分配唯一向量，无法处理一词多义 (如 “bank” 的河岸/银行)。

---

## PROBLEM 4 · 动态上下文感知嵌入

- **解决**：序列编码器 — 令词表示为整个输入序列的函数。

  - **架构**：**Transformer**

    - **自注意力 (Self‑Attention)**：第 $i$ 个 token 的表示 $\mathbf{h}_i$ 是序列中所有 token 的值向量加权和：

$$
\mathbf{h}_i = \sum_j \alpha_{ij} \mathbf{v}_j, \quad
\alpha_{ij} = \text{softmax}_j \left( \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} \right)
$$
    - 多层叠加，最终产生上下文相关嵌入 (Contextual Embeddings) $\mathbf{h}_i^{(L)}$。
  - **训练任务**：掩码语言模型 (MLM)

    - 随机遮蔽部分 token，根据周围未遮蔽上下文预测被遮蔽的原始词，迫使模型融合双向上下文。
- **结果**：同一词在不同上下文中获得不同嵌入，语义消歧天然完成。
- **新问题** → 嵌入已超越“词”的范畴。图节点、用户、商品如何共享同一套嵌入原理？

---

## PROBLEM 5 · 嵌入的通用原理抽象

- **通用定义**：  
**Embedding** 是一种学得的映射，将高维稀疏的离散实体投影到低维流形，使实体间的交互 (共现、连接、点击) 可通过内积或距离度量重建。
- **统一数学框架**：**对比学习 (Contrastive Learning)**

  - 给定正样本对 $(a, p)$ 和一组负样本 $\{n_i\}$，目标是最小化：

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(a, p)/\tau)}{\exp(\text{sim}(a, p)/\tau) + \sum_i \exp(\text{sim}(a, n_i)/\tau)}
$$

    - $\text{sim}(\mathbf{u},\mathbf{v})$ 通常为余弦相似度 (即 L2 归一化后的内积)。
    - 温度系数 $\tau$ 控制分布锐度。
  - **本质**：拉近正样本，推开负样本，使向量空间的组织结构反映领域本体或行为相似性。
- **最终推论**：  
任何可定义“正样本关系 (同现/邻居/用户点击)”的离散系统，均可通过上述对比损失学习 **Embedding**。所得低维连续向量完成了对原始高维结构的**语义压缩**与**流形展开**，使得代数运算等价于推理。

---

## SEE ALSO

`word2vec(1)`, `transformer(7)`, `contrastive-learning(7)`, `representation-learning(7)`