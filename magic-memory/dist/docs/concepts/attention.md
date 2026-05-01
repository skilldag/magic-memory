---
id: attention
title: 注意力机制
alias: [Attention, QKV]
level: 2
category: vLLM
depends_on: [vllm-tree/transformer.md]
leads_to: [level-2/paged-attention]
related: [level-2/multi-head-attention]
problem: "如何让模型知道重点关注哪些词?"
gap_anticipate: "Q/K/V是什么?为什么需要三个?"
---

# 注意力机制

## 问题驱动

Transformer处理序列时，所有词是并行处理的。但模型需要知道"重点关注哪些词"——这就是注意力机制要解决的问题。

## 类比理解

像查字典：
- **Q (Query)**: 你想查的词
- **K (Key)**: 词典里的词  
- **V (Value)**: 词的定义

你用Q去匹配K，得到对应的V的权重。

## 为什么需要三个?

分离Q/K/V可以让模型学习不同的匹配策略，增加表达能力。

## 衍生概念

- **PagedAttention**: 分页管理的注意力
- **FlashAttention**: 快速的注意力实现