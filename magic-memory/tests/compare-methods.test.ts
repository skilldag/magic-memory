/**
 * 五种知识点提取方法对比测试
 * 运行: npx tsx tests/compare-methods.test.ts
 */

import { buildConceptGraphFromText, textrank, positionWeighted, fusionExtract, structuralExtract } from '../src/utils/alignment'
import type { Concept } from '../src/types'

const TEXT = `# 注意力机制

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

const CONCEPTS: Concept[] = [
  { id: 'concepts/attention.md', title: '注意力机制', alias: ['Attention', 'QKV', 'attention'], level: 2, category: 'vLLM', problem: '如何让模型知道重点关注哪些词?', gap_anticipate: 'Q/K/V是什么?为什么需要三个?', depends_on: [], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
]

function sep(title: string) {
  console.log(`\n${'━'.repeat(60)}\n${title}\n${'━'.repeat(60)}`)
}

// ========== 方法 1: 社区发现（现有） ==========

sep('方法 1: 社区发现（现有方法）')
const m1 = buildConceptGraphFromText(TEXT, CONCEPTS)
m1.nodeGroups.forEach((ng, i) => {
  console.log(`  ${i + 1}. ${ng.label}${ng.isKnownConcept ? ' [KG]' : ''}`)
})

// ========== 方法 2: TextRank ==========

sep('方法 2: TextRank（PageRank 图排序）')
const m2 = textrank(TEXT, CONCEPTS)
m2.slice(0, 15).forEach((t, i) => {
  console.log(`  ${i + 1}. ${t.term}  (score: ${t.score.toFixed(4)})`)
})

// ========== 方法 3: 位置加权 ==========

sep('方法 3: 位置加权')
const m3 = positionWeighted(TEXT, CONCEPTS)
m3.slice(0, 15).forEach((t, i) => {
  console.log(`  ${i + 1}. ${t.term}  (score: ${t.score.toFixed(1)})`)
})

// ========== 方法 4: 融合策略 ==========

sep('方法 4: 融合策略')
const m4 = fusionExtract(TEXT, CONCEPTS)
m4.nodeGroups.forEach((ng, i) => {
  const scores = ng.terms
    .map(t => ({ term: t, score: m2.find(x => x.term.toLowerCase() === t.toLowerCase())?.score ?? 0 }))
    .sort((a, b) => b.score - a.score)
  console.log(`  ${i + 1}. ${ng.label}${ng.isKnownConcept ? ' [KG]' : ''}`)
  if (scores.length > 1) {
    console.log(`     terms: ${scores.slice(0, 3).map(s => `${s.term}(${s.score.toFixed(4)})`).join(', ')}`)
  }
})

// ========== 方法 5: StructuralRank（新算法） ==========

sep('方法 5: StructuralRank（新算法）')
const m5 = structuralExtract(TEXT, CONCEPTS)
m5.nodeGroups.forEach((ng, i) => {
  console.log(`  ${i + 1}. ${ng.label}${ng.isKnownConcept ? ' [KG]' : ''}`)
})

// ========== 对比总结 ==========

sep('对比总结')
console.log('')
console.log('  方法              知识点数  类型')
console.log('  ─────────────────────────────────')
console.log(`  1.社区发现         ${m1.nodeGroups.length}        分组/聚类`)
console.log(`  2.TextRank(PR)     ${m2.length}        排序列表`)
console.log(`  3.位置加权         ${m3.length}        排序列表`)
console.log(`  4.融合策略         ${m4.nodeGroups.length}        分组+排序`)
console.log(`  5.StructuralRank   ${m5.nodeGroups.length}        分组+位置标签`)

console.log('\n  社区发现:', m1.nodeGroups.map(g => g.label).join(', '))
console.log('  TextRank Top 5:', m2.slice(0, 5).map(t => t.term).join(', '))
console.log('  位置加权 Top 5:', m3.slice(0, 5).map(t => t.term).join(', '))
console.log('  融合策略:', m4.nodeGroups.map(g => g.label).join(', '))
console.log('  StructuralRank:', m5.nodeGroups.map(g => g.label).join(', '))
