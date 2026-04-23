# Speculative Decoding 深层推导

> 从"自回归如何变推测"往下推，推到不能再分

---

## 表层问题

**输入**：已生成 tokens [t1, t2, t3]
**输出**：新 token tn+1
**问题**：自回归太慢，如何一次多生成几个？

---

## 第一层推导

```
    [自回归 → 推测]
          │
    ┌─────┴─────┐
    │           │
[起草] [验证]
    │           │
    ▼           ▼
[Proposer] [Verifier]
    │           │
    └─────┬─────┘
          │
    [Speculative Decoding]
```

| 层级 | 问题 | 解决 | 概念 |
|------|------|------|------|
| 表层 | 如何加速自回归？ | 推测 | Speculative Decoding |
| 底层 | 如何起草？ | N-gram/小模型 | Proposer |
| 底层 | 如何验证？ | 大模型验证 | Verifier |
| 底层 | 如何接受？ | 拒绝采样 | Acceptance |

---

## 第二层推导：Proposer

### 2.1 N-gram Proposer

```
[N-gram]
    │
    ├── 问题：如何预测？
    │       解决：查表
    │
    ├── 公式：P(t) = count(t-n+1...t) / count(t-n+1...t-1)
    │
    └── 优点：快
            缺点：质量低
```

### 2.2 小模型 Proposer

```
[Small Model]
    │
    ├── 问题：用什么模型？
    │       解决：1-3B的小模型
    │
    ├── 问题：如何训练？
    │       解决：蒸馏
    │
    └── 问题：如何协同？
            解决：共享vocab
```

---

## 第三层推导：Verification & Acceptance

### 3.1 验证流程

```
[Verification]
    │
    ├── 1. Proposer 生成 k 个 token
    │           [t1', t2', ..., tk']
    │
    ├── 2. 大模型逐个验证
    │           计算 p_orig(ti')
    │           计算 p_prop(ti')
    │
    └── 3. 计算接受概率
            accept_prob = min(1, p_orig/p_prop)
```

### 3.2 接受策略

```
[Acceptance]
    │
    ├── 逐个检查：
    │       │
    │       ├── 接受：继续
    │       │
    │       └── 拒绝：停止，返回
    │
    └── 最终输出：
            接受的前n个
            大模型生成的第n+1个
```

### 3.3 终止检查

```
[Termination]
    │
    ├── 问题：何时停止？
    │       解决：终止token或达长度
    │
    └── 问题：终止token被拒绝？
            解决：替换为终止
```

---

## 完整推导树

```
Speculative Decoding
    │
    ├── [Proposer]
    │       ├── N-gram Lookup
    │       │       ├── trigram
    │       │       └── bigram
    │       │
    │       └── Small Model
    │               ├── 蒸馏
    │               └── 共享vocab
    │
    ├── [Verifier]
    │       ├── 大模型验证
    │       ├── 概率比较
    │       └── 条件接受
    │
    └── [Acceptance]
            ├── 逐个接受
            ├── 终止检查
            └── 回退机制
```

---

## 记住

```
Speculative Decoding = Proposer(快) + Verifier(准) + Acceptance(控制)
              │
              └── 核心问题：如何用小成本换大收益？
```