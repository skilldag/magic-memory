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

// ========== KEY CONCEPTS 解析 ==========

function parseKeyConcepts(text: string): string[] {
  const lines = text.split('\n')
  let inSection = false
  const concepts: string[] = []
  for (const line of lines) {
    const trimmed = line.trim()
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
  return concepts
}

function charJaccard(a: string, b: string): number {
  const sa = new Set(a); const sb = new Set(b)
  const inter = [...sa].filter(c => sb.has(c)).length
  const union = new Set([...sa, ...sb]).size
  return union > 0 ? inter / union : 0
}

/** 简易分词：从文本中提取有意义的术语（英文词 + 中文短语），用于 extra 检测 */
function extractTerms(text: string): string[] {
  const tokens = new Set<string>()

  // 英文词：字母开头，2-30 字符
  for (const w of text.split(/[\s,，、；;：:()（）\[\]【】「」""''，。！!？?\n\r]+/)) {
    const clean = w.replace(/^[\s#\-*]+/, '').trim()
    if (/^[a-zA-Z][a-zA-Z0-9-_]{1,30}$/.test(clean) && clean.length >= 2) {
      tokens.add(clean)
    }
  }

  // 中文短语：2+ 个连续汉字（取消原来的 8 字上限）
  const segments = text.split(/[，,。、；;：:()（）\[\]【】「」""''\s\n\r]+/)
  for (const seg of segments) {
    const clean = seg.replace(/[^一-龥]/g, '')
    if (clean.length >= 2) {
      tokens.add(clean)
    }
  }

  return [...tokens]
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

// ========== 对外接口 ==========

/** 模糊匹配阈值：charJaccard 高于此值视为语义近似 */
const FUZZY_THRESHOLD = 0.55

export function compareTexts(
  userText: string,
  originalContent: string,
  allConcepts: Concept[],
  subjectConceptId?: string
): GraphAlignmentResult {
  // 1. 提取 KEY CONCEPTS
  const keyConcepts = parseKeyConcepts(originalContent)

  // 没有 KEY CONCEPTS 时，回退到老的分词方案
  if (keyConcepts.length === 0) {
    const userTokens = extractTerms(userText)
    const originalTokens = extractTerms(originalContent)
    return alignTokenSets(userTokens, originalTokens)
  }

  // 2. 先提取用户输入中的术语，后面多次复用
  const userTerms = extractTerms(userText)

  // 3. 匹配：精确子串匹配 + charJaccard 模糊近似
  const matchedLabels = new Set<string>()
  const matchedNodes: AlignedNodePair[] = []
  const missingNodes: AlignedNodePair[] = []
  // 记录哪些 keyConcepts 是模糊匹配上的，用于 fuzzyMatches 报告
  const fuzzyMatchedKc = new Map<string, { userTerm: string; score: number }>()

  for (let i = 0; i < keyConcepts.length; i++) {
    const kc = keyConcepts[i]
    const isEnglish = kc.length > 0 && kc.charCodeAt(0) < 128
    let found: boolean

    if (isEnglish) {
      found = new RegExp(escapeRegex(kc), 'i').test(userText)
    } else {
      // 精确子串匹配（快路径）
      found = userText.includes(kc)
      if (!found && kc.length >= 2) {
        // 模糊近似：在用户术语中找 charJaccard 最高的
        let bestScore = 0
        let bestTerm = ''
        for (const ut of userTerms) {
          const score = charJaccard(kc, ut)
          if (score > bestScore) {
            bestScore = score
            bestTerm = ut
          }
        }
        if (bestScore >= FUZZY_THRESHOLD) {
          found = true
          fuzzyMatchedKc.set(kc, { userTerm: bestTerm, score: bestScore })
        }
      }
    }

    const node: AlignedNodePair = {
      nodeId: `kc_${i}`,
      label: kc,
      isKnownConcept: false,
      inUser: found,
      inOriginal: true,
      status: found ? 'matched' : 'missing',
    }

    if (found) {
      matchedLabels.add(kc)
      matchedNodes.push(node)
    } else {
      missingNodes.push(node)
    }
  }

  // 4. Extra：用户术语中未被 KEY CONCEPTS 覆盖的（精确 + 模糊都算覆盖）
  const extraTerms = userTerms.filter(t =>
    !keyConcepts.some(kc =>
      t.includes(kc) || kc.includes(t) || charJaccard(kc, t) >= FUZZY_THRESHOLD
    )
  )
  const uniqueExtra = [...new Set(extraTerms)]
  const extraNodes: AlignedNodePair[] = uniqueExtra.map((label, i) => ({
    nodeId: `extra_${i}`,
    label,
    isKnownConcept: false,
    inUser: true,
    inOriginal: false,
    status: 'extra',
  }))

  const allNodes = [...matchedNodes, ...missingNodes, ...extraNodes]

  // 5. 构建模糊匹配报告
  const fuzzyMatches: { userLabel: string; originalLabel: string; similarity: number }[] = []
  // 5a. 模糊匹配上的 keyConcepts
  for (const [kc, match] of fuzzyMatchedKc) {
    fuzzyMatches.push({
      userLabel: match.userTerm,
      originalLabel: kc,
      similarity: Math.round(match.score * 100),
    })
  }
  // 5b. 遗留：extra 对 missing 的 charJaccard 提示（阈值降低到 0.3，作为弱提示）
  const extraLabels = extraNodes.map(n => n.label)
  const missingLabels = missingNodes.map(n => n.label)
  for (const ex of extraLabels) {
    let best = ''; let bestSim = 0
    for (const ms of missingLabels) {
      const sim = charJaccard(ex, ms)
      if (sim > bestSim) { bestSim = sim; best = ms }
    }
    if (bestSim >= 0.3) {
      fuzzyMatches.push({ userLabel: ex, originalLabel: best, similarity: Math.round(bestSim * 100) })
    }
  }

  return {
    nodes: allNodes,
    edges: [],
    fuzzyMatches,
    userNodeCount: userTerms.length,
    originalNodeCount: keyConcepts.length,
    stats: {
      nodeCoverage: keyConcepts.length > 0
        ? Math.round((matchedNodes.length / keyConcepts.length) * 100) : 0,
      nodePrecision: allNodes.length > 0
        ? Math.round((matchedNodes.length / (matchedNodes.length + uniqueExtra.length)) * 100) : 0,
      edgeCoverage: 100,
      matchedNodeCount: matchedNodes.length,
      missingNodeCount: missingNodes.length,
      extraNodeCount: uniqueExtra.length,
      matchedEdgeCount: 0,
      missingEdgeCount: 0,
      extraEdgeCount: 0,
    },
  }
}

/** 回退方案：无 KEY CONCEPTS 时，做两个术语集合的交集比对 */
function alignTokenSets(userTokens: string[], originalTokens: string[]): GraphAlignmentResult {
  const matched: string[] = []
  const missing: string[] = []
  const matchedSet = new Set<string>()

  for (const ot of originalTokens) {
    const found = userTokens.some(ut => ut.includes(ot) || ot.includes(ut))
    if (found) {
      matched.push(ot)
      matchedSet.add(ot)
    } else {
      missing.push(ot)
    }
  }

  const extra = userTokens.filter(t => !matchedSet.has(t) && !originalTokens.some(ot => ot.includes(t) || t.includes(ot)))

  const nodes: AlignedNodePair[] = [
    ...matched.map((l, i) => ({ nodeId: `t_${i}`, label: l, isKnownConcept: false, inUser: true, inOriginal: true, status: 'matched' as const })),
    ...missing.map((l, i) => ({ nodeId: `m_${i}`, label: l, isKnownConcept: false, inUser: false, inOriginal: true, status: 'missing' as const })),
    ...extra.map((l, i) => ({ nodeId: `e_${i}`, label: l, isKnownConcept: false, inUser: true, inOriginal: false, status: 'extra' as const })),
  ]

  return {
    nodes,
    edges: [],
    fuzzyMatches: [],
    userNodeCount: userTokens.length,
    originalNodeCount: originalTokens.length,
    stats: {
      nodeCoverage: originalTokens.length > 0 ? Math.round((matched.length / originalTokens.length) * 100) : 0,
      nodePrecision: userTokens.length > 0 ? Math.round((matched.length / userTokens.length) * 100) : 0,
      edgeCoverage: 100,
      matchedNodeCount: matched.length,
      missingNodeCount: missing.length,
      extraNodeCount: extra.length,
      matchedEdgeCount: 0, missingEdgeCount: 0, extraEdgeCount: 0,
    },
  }
}

export function removeKeyConceptFromContent(
  content: string,
  termToRemove: string
): string | null {
  const lines = content.split('\n')
  let inSection = false
  let modified = false

  const result = lines.map(line => {
    const trimmed = line.trim()
    if (/^#*\s*KEY CONCEPTS:?\s*$/i.test(trimmed)) {
      inSection = true
      return line
    }
    if (inSection) {
      if (trimmed.startsWith('#') || (trimmed === '' && modified)) {
        inSection = false
        return line
      }
      if (trimmed) {
        const terms = line.split(/\s+/)
        const filtered = terms.filter(t => t !== termToRemove)
        if (filtered.length !== terms.length) {
          modified = true
        }
        return filtered.join(' ')
      }
    }
    return line
  })

  return modified ? result.join('\n') : null
}
