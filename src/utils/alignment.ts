import type { Concept } from '../types'

// ========== 图类型 ==========

export interface GraphNode {
  id: string
  label: string
  isKnownConcept: boolean
  conceptId?: string
  terms: string[]
}

export interface GraphEdge {
  sourceId: string
  targetId: string
  weight: number
}

export interface TextGraph {
  nodeGroups: GraphNode[]
  edges: GraphEdge[]
}

export interface AlignedNodePair {
  nodeId: string
  label: string
  isKnownConcept: boolean
  inUser: boolean
  inOriginal: boolean
  status: 'matched' | 'missing' | 'extra'
}

export interface AlignedEdgePair {
  sourceId: string
  targetId: string
  sourceLabel: string
  targetLabel: string
  inUser: boolean
  inOriginal: boolean
  status: 'matched' | 'missing' | 'extra'
}

export interface GraphAlignmentResult {
  nodes: AlignedNodePair[]
  edges: AlignedEdgePair[]
  userNodeCount: number
  originalNodeCount: number
  fuzzyMatches: { userLabel: string; originalLabel: string; similarity: number }[]
  stats: {
    nodeCoverage: number
    nodePrecision: number
    edgeCoverage: number
    matchedNodeCount: number
    missingNodeCount: number
    extraNodeCount: number
    matchedEdgeCount: number
    missingEdgeCount: number
    extraEdgeCount: number
  }
}

// ========== 简易 Tokenizer（替代被移除的 NLP 管道） ==========

/** 对用户输入做极简分词：保留英文技术词和中文短语 */
function tokenize(text: string): string[] {
  const tokens = new Set<string>()

  // 英文词：字母开头，2-30 字符
  for (const w of text.split(/[\s,，、；;：:()（）\[\]【】「」""''，。！!？?\n\r]+/)) {
    const clean = w.replace(/^[\s#\-*]+/, '').trim()
    if (/^[a-zA-Z][a-zA-Z0-9-_]{1,30}$/.test(clean) && clean.length >= 2) {
      tokens.add(clean)
    }
  }

  // 中文短语：2-8 个连续汉字
  const segments = text.split(/[，,。、；;：:()（）\[\]【】「」""''\s\n\r]+/)
  for (const seg of segments) {
    const clean = seg.replace(/[^一-龥]/g, '')
    if (clean.length >= 2 && clean.length <= 8) {
      tokens.add(clean)
    }
  }

  return [...tokens]
}

function buildSimpleGraph(tokens: string[]): TextGraph {
  const nodeGroups: GraphNode[] = tokens.map((t, i) => ({
    id: `t_${i}`,
    label: t,
    isKnownConcept: false,
    terms: [t],
  }))
  const edges: GraphEdge[] = []
  for (let i = 0; i < tokens.length; i++) {
    for (let j = i + 1; j < tokens.length; j++) {
      edges.push({ sourceId: `t_${i}`, targetId: `t_${j}`, weight: 1 })
    }
  }
  return { nodeGroups, edges }
}

// ========== KEY CONCEPTS 解析 ==========

function parseKeyConcepts(text: string): string[] {
  const lines = text.split('\n')
  let inSection = false
  const concepts: string[] = []
  for (const line of lines) {
    const trimmed = line.trim()
    // 匹配 KEY CONCEPTS（支持 ## KEY CONCEPTS、KEY CONCEPTS:、大小写不敏感）
    if (/^#*\s*KEY CONCEPTS:?\s*$/i.test(trimmed)) {
      inSection = true
      continue
    }
    if (inSection) {
      if (trimmed.startsWith('#') || (trimmed === '' && concepts.length > 0)) break
      if (trimmed) {
        concepts.push(...trimmed.split(/\s+/))
      }
    }
  }
  // KEY CONCEPTS 是人工标注的，不过滤长度
  return concepts
}

function buildKeyConceptGraph(keys: string[]): TextGraph {
  const nodeGroups: GraphNode[] = keys.map((key, i) => ({
    id: `kc_${i}`,
    label: key,
    isKnownConcept: false,
    terms: [key],
  }))
  const edges: GraphEdge[] = []
  for (let i = 0; i < keys.length; i++) {
    for (let j = i + 1; j < keys.length; j++) {
      edges.push({ sourceId: `kc_${i}`, targetId: `kc_${j}`, weight: 1 })
    }
  }
  return { nodeGroups, edges }
}

// ========== 图对齐 ==========

function charJaccard(a: string, b: string): number {
  const sa = new Set(a); const sb = new Set(b)
  const inter = [...sa].filter(c => sb.has(c)).length
  const union = new Set([...sa, ...sb]).size
  return union > 0 ? inter / union : 0
}

export function alignGraphs(
  userGraph: TextGraph,
  originalGraph: TextGraph
): GraphAlignmentResult {
  const uMap = new Map(userGraph.nodeGroups.map(n => [n.id, n]))
  const oMap = new Map(originalGraph.nodeGroups.map(n => [n.id, n]))

  // 基于 shared terms 匹配
  const matchedU = new Set<string>()
  const matchedO = new Set<string>()

  for (const [uid, un] of uMap) {
    let bestO = ''; let bestScore = 0
    for (const [oid, on] of oMap) {
      const overlap = un.terms.filter(t => on.terms.includes(t)).length
      if (overlap > bestScore) { bestScore = overlap; bestO = oid }
    }
    if (bestScore === 0) {
      for (const [oid, on] of oMap) {
        const sim = charJaccard(un.label, on.label)
        if (sim > 0.3 && sim > bestScore) { bestScore = sim; bestO = oid }
      }
    }
    if (bestScore > 0) { matchedU.add(uid); matchedO.add(bestO) }
  }

  const nodes: AlignedNodePair[] = []
  for (const [uid, un] of uMap) {
    nodes.push({ nodeId: uid, label: un.label, isKnownConcept: un.isKnownConcept, inUser: true, inOriginal: matchedU.has(uid), status: matchedU.has(uid) ? 'matched' : 'extra' })
  }
  for (const [oid, on] of oMap) {
    if (!matchedO.has(oid)) {
      nodes.push({ nodeId: oid, label: on.label, isKnownConcept: on.isKnownConcept, inUser: false, inOriginal: true, status: 'missing' })
    }
  }

  // 模糊匹配
  const extra = nodes.filter(n => n.status === 'extra')
  const missing = nodes.filter(n => n.status === 'missing')
  const fuzzy: { userLabel: string; originalLabel: string; similarity: number }[] = []
  for (const ex of extra) {
    let best = ''; let bestSim = 0
    for (const ms of missing) {
      const sim = charJaccard(ex.label, ms.label)
      if (sim > bestSim) { bestSim = sim; best = ms.label }
    }
    if (bestSim >= 0.3) fuzzy.push({ userLabel: ex.label, originalLabel: best, similarity: Math.round(bestSim * 100) })
  }

  // 边对齐
  const uEk = new Set(userGraph.edges.map(e => [e.sourceId, e.targetId].sort().join('--')))
  const oEk = new Set(originalGraph.edges.map(e => [e.sourceId, e.targetId].sort().join('--')))
  const aEk = new Set([...uEk, ...oEk])
  const edges: AlignedEdgePair[] = []
  for (const ek of aEk) {
    const [s, t] = ek.split('--')
    const iu = uEk.has(ek); const io = oEk.has(ek)
    edges.push({
      sourceId: s, targetId: t,
      sourceLabel: nodes.find(n => n.nodeId === s)?.label ?? s,
      targetLabel: nodes.find(n => n.nodeId === t)?.label ?? t,
      inUser: iu, inOriginal: io,
      status: iu && io ? 'matched' : io ? 'missing' : 'extra',
    })
  }

  const mn = nodes.filter(n => n.status === 'matched')
  const ms = nodes.filter(n => n.status === 'missing')
  const ex = nodes.filter(n => n.status === 'extra')

  return {
    nodes, edges, fuzzyMatches: fuzzy,
    userNodeCount: uMap.size, originalNodeCount: oMap.size,
    stats: {
      nodeCoverage: oMap.size > 0 ? Math.round((mn.length / oMap.size) * 100) : 0,
      nodePrecision: uMap.size > 0 ? Math.round((mn.length / uMap.size) * 100) : 0,
      edgeCoverage: oEk.size > 0 ? Math.round((mn.length / oEk.size) * 100) : 100,
      matchedNodeCount: mn.length, missingNodeCount: ms.length, extraNodeCount: ex.length,
      matchedEdgeCount: 0, missingEdgeCount: 0, extraEdgeCount: 0,
    },
  }
}

// ========== 对外接口 ==========

export function compareTexts(
  userText: string,
  originalContent: string,
  allConcepts: Concept[],
  subjectConceptId?: string
): GraphAlignmentResult {
  // 用户输入：简易分词
  const userTokens = tokenize(userText)
  const userGraph = buildSimpleGraph(userTokens)

  // 原文：KEY CONCEPTS 段落
  const keyConcepts = parseKeyConcepts(originalContent)
  const originalGraph = keyConcepts.length > 0
    ? buildKeyConceptGraph(keyConcepts)
    : buildSimpleGraph(tokenize(originalContent))

  return alignGraphs(userGraph, originalGraph)
}
