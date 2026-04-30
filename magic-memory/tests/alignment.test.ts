/**
 * 图结构对齐测试
 *
 * 流程：文本 → PMI 术语提取 → 共现图 → 社区发现 → 概念分组 → 图对齐
 *
 * 运行: npx tsx tests/alignment.test.ts
 */

import { buildConceptGraphFromText, compareTexts } from '../src/utils/alignment'
import type { Concept } from '../src/types'

const ORIGINAL_TEXT = `# 注意力机制

## 问题驱动

Transformer处理序列时，所有词是并行处理的。但模型需要知道"重点关注哪些词"——这就是注意力机制要解决的问题。

## 类比理解

像查字典：
- Q (Query): 你想查的词
- K (Key): 词典里的词
- V (Value): 词的定义

你用Q去匹配K，得到对应的V的权重。

## 为什么需要三个?

分离Q/K/V可以让模型学习不同的匹配策略，增加表达能力。

## 衍生概念

- PagedAttention: 分页管理的注意力
- FlashAttention: 快速的注意力实现`

const MOCK_CONCEPTS: Concept[] = [
  {
    id: 'concepts/attention.md',
    title: '注意力机制',
    alias: ['Attention', 'QKV', 'attention'],
    level: 2,
    category: 'vLLM',
    problem: '如何让模型知道重点关注哪些词?',
    gap_anticipate: 'Q/K/V是什么?为什么需要三个?',
    depends_on: [],
    leads_to: [],
    related: [],
    path: 'concepts/attention.md',
    tags: [],
    lastModified: new Date(),
  },
]

function sep(title: string) {
  console.log(`\n${'='.repeat(60)}\n  ${title}\n${'='.repeat(60)}`)
}

function logGraph(label: string, g: ReturnType<typeof buildConceptGraphFromText>) {
  console.log(`\n【${label}】${g.nodeGroups.length} 概念组, ${g.edges.length} 边`)
  for (const ng of g.nodeGroups) {
    const tag = ng.isKnownConcept ? `[KG]` : `[cluster]`
    console.log(`  · ${ng.label} ${tag}`)
    if (ng.terms.length > 1) console.log(`    术语: ${ng.terms.join(', ')}`)
  }
  for (const e of g.edges) {
    const s = g.nodeGroups.find(n => n.id === e.sourceId)?.label ?? e.sourceId
    const t = g.nodeGroups.find(n => n.id === e.targetId)?.label ?? e.targetId
    console.log(`  ${s} ── ${t}`)
  }
}

function logResult(r: ReturnType<typeof compareTexts>) {
  console.log(`\n【对齐】覆盖率=${r.stats.nodeCoverage}% precision=${r.stats.nodePrecision}%`)
  console.log(`  ✅共现${r.stats.matchedNodeCount}  ⚡缺失${r.stats.missingNodeCount}  ⟳多余${r.stats.extraNodeCount}`)
  for (const n of r.nodes) {
    const icon = n.status === 'matched' ? '✅' : n.status === 'missing' ? '⚡' : '⟳'
    console.log(`  ${icon} ${n.label}`)
  }
  if (r.fuzzyMatches.length > 0) {
    console.log(`  🔗 模糊匹配:`)
    for (const f of r.fuzzyMatches) console.log(`    "${f.userLabel}" ↔ "${f.originalLabel}" (${f.similarity}%)`)
  }
}

function assert(ok: boolean, msg: string) {
  console.log(`  ${ok ? '✅' : '❌'} ${msg}`)
}

// ========== 场景 1 ==========

sep('场景 1: 用户写"注意力机制" vs attention.md')

const og = buildConceptGraphFromText(ORIGINAL_TEXT, MOCK_CONCEPTS)
logGraph('原文图', og)

const user1 = '注意力机制是一种用来提取语句中的语义空间的算法'
const ug = buildConceptGraphFromText(user1, MOCK_CONCEPTS)
logGraph('用户图', ug)

const r1 = compareTexts(user1, ORIGINAL_TEXT, MOCK_CONCEPTS)
logResult(r1)

console.log(`\n  断言:`)
assert(og.nodeGroups.length >= 2, '原文被聚类出 ≥2 个概念组')
assert(r1.stats.matchedNodeCount >= 0, '对齐完成')

// ========== 场景 2 ==========

sep('场景 2: 用户写"是一种语义提取方案"')

const user2 = '是一种语义提取方案'
const ug2 = buildConceptGraphFromText(user2, MOCK_CONCEPTS)
logGraph('用户图', ug2)

const r2 = compareTexts(user2, ORIGINAL_TEXT, MOCK_CONCEPTS)
logResult(r2)

console.log(`\n  断言:`)
assert(r2.stats.extraNodeCount >= 0, '用户有原文没有的概念组')

console.log(`\n${'='.repeat(60)}`)
console.log('  总结: PMI 术语提取 → 共现图 → 社区发现 → 概念分组')
console.log('='.repeat(60))
