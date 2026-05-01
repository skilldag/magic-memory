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

// ========== Step 1: PMI-based Chinese Term Extraction ==========

const STOP = new Set('的了一是在不也有大这中人上为所如把被让给对从到要和会可以但因为所以如果虽然然而而且或者之后前能没很又再才')

/** 去掉 markdown 工件（代码块、ASCII图、emoji、链接等） */
function scrub(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/!\[.*?\]\(.*?\)/g, ' ')
    .replace(/\[.*?\]\(.*?\)/g, ' ')
    .replace(/[─│┌┐└┘├┤┬┴┼▼▲→←▸▪●○■□➤📍💡❓❗➡️✨🔍📉📌🧠⚡]+/g, ' ')
    .replace(/^[│├└┌┤▸▪\s─┐┘┬┴┼▼▲◀▶]+$/gm, '')
}

function extractTerms(raw: string): string[] {
  const text = scrub(raw)
  const candidates = new Map<string, number>()

  // 1. 中文标点分隔短语
  const segments = text.split(/[，,。、；;：:()（）\[\]【】「」""''\s\n\r]+/)
  for (const seg of segments) {
    const clean = seg.replace(/[^一-龥]/g, '')
    if (clean.length < 2) continue
    if (!STOP.has(clean[clean.length - 1])) {
      if (clean.length <= 8) {
        candidates.set(clean, (candidates.get(clean) ?? 0) + 1)
      } else {
        for (let win = 6; win >= 4; win--) {
          for (let i = 0; i <= clean.length - win; i++) {
            const sub = clean.slice(i, i + win)
            if (!STOP.has(sub[sub.length - 1])) {
              candidates.set(sub, (candidates.get(sub) ?? 0) + 1)
            }
          }
        }
      }
    }
  }

  // 2. 英文技术术语（概念多为英文）
  const words = text.split(/[\s,，、；;：:()（）\[\]【】「」""''，。！!？?\n\r]+/)
  for (const w of words) {
    const clean = w.replace(/^[\s#\-*]+/, '').trim()
    if (/^[a-zA-Z][a-zA-Z0-9-_]{1,30}$/.test(clean) && clean.length >= 2) {
      candidates.set(clean, (candidates.get(clean) ?? 0) + 1)
    }
  }

  // 3. 标题行和列表项（冒号前部分）
  const lines = text.split('\n')
  for (const line of lines) {
    const title = line.replace(/^[\s#\-*]+/, '').replace(/[：:].*$/, '').trim()
    if (title.length >= 2 && title.length <= 12) {
      candidates.set(title, (candidates.get(title) ?? 0) + 1)
    }
  }

  return [...candidates.keys()]
}

// ========== Step 2: Community Detection (Label Propagation) ==========

function detectCommunities(
  nodeIds: string[],
  adjList: Map<string, Set<string>>
): Map<string, number> {
  const labels = new Map<string, number>()
  nodeIds.forEach((id, i) => labels.set(id, i))

  for (let iter = 0; iter < 30; iter++) {
    let changed = false
    const order = [...nodeIds].sort(() => Math.random() - 0.5)
    for (const id of order) {
      const neighbors = adjList.get(id)
      if (!neighbors || neighbors.size === 0) continue
      const count = new Map<number, number>()
      for (const n of neighbors) {
        const l = labels.get(n)
        if (l !== undefined) count.set(l, (count.get(l) ?? 0) + 1)
      }
      if (count.size === 0) continue
      let best = labels.get(id)!
      let bestCount = 0
      for (const [l, c] of count) {
        if (c > bestCount) { bestCount = c; best = l }
      }
      if (labels.get(id) !== best) { labels.set(id, best); changed = true }
    }
    if (!changed) break
  }
  return labels
}

// ========== Step 3: Full Pipeline ==========

export function buildConceptGraphFromText(
  text: string,
  allConcepts: Concept[]
): TextGraph {
  const kgMap = new Map<string, { id: string; title: string }>()
  for (const c of allConcepts) {
    kgMap.set(c.title.toLowerCase(), { id: c.id, title: c.title })
    if (c.alias) {
      for (const a of c.alias) kgMap.set(a.toLowerCase(), { id: c.id, title: c.title })
    }
  }
  const sortedKg = [...kgMap.entries()].sort((a, b) => b[0].length - a[0].length)

  const sentences = text.split(/[。！？.!?\n\r]+/).map(s => s.trim()).filter(s => s.length > 0)
  const terms = extractTerms(text)

  // 逐句提取 items
  const sentenceItems: Set<string>[] = []
  for (const sentence of sentences) {
    const lower = sentence.toLowerCase()
    const items = new Set<string>()
    const positions: { start: number; end: number }[] = []
    for (const [kw, info] of sortedKg) {
      let pos = 0
      while (true) {
        const idx = lower.indexOf(kw, pos)
        if (idx === -1) break
        if (!positions.some(p => idx < p.end && idx + kw.length > p.start)) {
          positions.push({ start: idx, end: idx + kw.length })
          items.add('kg:' + info.id)
          pos = idx + kw.length
        } else { pos = idx + 1 }
      }
    }
    for (const t of terms) {
      if (lower.includes(t.toLowerCase())) items.add('t:' + t)
    }
    if (items.size > 0) sentenceItems.push(items)
  }

  // 建共现邻接表
  const nodeSet = new Set<string>()
  const adjList = new Map<string, Set<string>>()
  for (const items of sentenceItems) {
    const arr = [...items]
    for (const item of arr) nodeSet.add(item)
    for (let i = 0; i < arr.length; i++) {
      for (let j = i + 1; j < arr.length; j++) {
        if (!adjList.has(arr[i])) adjList.set(arr[i], new Set())
        if (!adjList.has(arr[j])) adjList.set(arr[j], new Set())
        adjList.get(arr[i])!.add(arr[j])
        adjList.get(arr[j])!.add(arr[i])
      }
    }
  }
  for (const items of sentenceItems) {
    for (const item of items) {
      if (!adjList.has(item)) adjList.set(item, new Set())
    }
  }

  const allNodes = [...nodeSet]

  // 社区发现
  const communities = detectCommunities(allNodes, adjList)

  // 合并社区
  const commItems = new Map<number, Set<string>>()
  for (const [id, cid] of communities) {
    if (!commItems.has(cid)) commItems.set(cid, new Set())
    commItems.get(cid)!.add(id)
  }

  const nodeGroups: GraphNode[] = []
  const itemToGroup = new Map<string, string>()

  // 从原文中预扫描"概念: 描述"模式和标题行，提取概念候选
  const conceptCandidates = new Map<string, number>()
  const colonPatterns = text.match(/[^,，\n]+[:：][^,，\n]+/g)
  if (colonPatterns) {
    for (const p of colonPatterns) {
      const left = p.split(/[:：]/)[0].replace(/^[\s#\-*]+/, '').trim()
      if (left.length >= 2 && left.length <= 30) {
        const bonus = /[a-zA-Z]/.test(left) ? 6 : 4
        conceptCandidates.set(left.toLowerCase(), bonus)
        const clean = left.replace(/[^\u4e00-\u9fff a-zA-Z0-9]/g, '').trim()
        if (clean !== left && clean.length >= 2) {
          conceptCandidates.set(clean.toLowerCase(), bonus)
        }
      }
    }
  }
  // 标题行也是概念候选
  for (const line of text.split('\n')) {
    const h = line.replace(/^[\s#]+/, '').replace(/[：:].*$/, '').trim()
    if (h.length >= 2 && h.length <= 20 && /[\u4e00-\u9fff]/.test(h)) {
      conceptCandidates.set(h.toLowerCase(), 3)
    }
  }
  // KG 概念名全局候选（即使文本中没有标题/冒号模式）
  for (const c of allConcepts) {
    conceptCandidates.set(c.title.toLowerCase(), 5)
    if (c.alias) {
      for (const a of c.alias) conceptCandidates.set(a.toLowerCase(), 5)
    }
  }

  // 停用动词——含这些词的不是概念而是描述
  const VERBS = new Set(['是', '需要', '处理', '学习', '匹配', '得到', '分离', '增加', '实现', '管理', '查', '驱动', '理解', '叫', '让', '可以', '用来'])

  function conceptScore(item: string, degree: number): number {
    const raw = item.startsWith('kg:') ? item.slice(3) : item.startsWith('t:') ? item.slice(2) : item
    const lower = raw.toLowerCase()
    let score = degree * 0.1
    if (item.startsWith('kg:')) score += 5
    if (conceptCandidates.has(lower)) score += conceptCandidates.get(lower)!
    if (raw.length >= 2 && raw.length <= 4) score += 2
    if (/[a-zA-Z]/.test(raw)) score += 2
    for (const v of VERBS) { if (raw.includes(v)) { score -= 3; break } }
    if (/^(为什么|什么|怎么|如何|是否)/.test(lower)) score -= 3
    return score
  }

  for (const [cid, items] of commItems) {
    let bestItem = ''
    let bestScore = -Infinity
    const allRaws = [...items].map(i => i.startsWith('kg:') ? i.slice(3) : i.startsWith('t:') ? i.slice(2) : i)

    for (const item of items) {
      const deg = adjList.get(item)?.size ?? 0
      let sc = conceptScore(item, deg)
      const raw = item.startsWith('kg:') ? item.slice(3) : item.startsWith('t:') ? item.slice(2) : item

      // 是其他术语的子串但未被概念候选收录？减分
      const isSubstr = allRaws.some(other => other !== raw && other.includes(raw))
      if (isSubstr && !conceptCandidates.has(raw.toLowerCase())) sc -= 3

      if (sc > bestScore) { bestScore = sc; bestItem = item }
    }

    const isKg = bestItem.startsWith('kg:')
    const kgId = isKg ? bestItem.slice(3) : undefined
    const c = kgId ? allConcepts.find(x => x.id === kgId) : undefined
    const label = isKg ? (c?.title ?? kgId!)
      : bestItem.startsWith('t:') ? bestItem.slice(2)
      : bestItem

    const allTerms = [...items].map(i =>
      i.startsWith('kg:') ? i.slice(3) : i.startsWith('t:') ? i.slice(2) : i
    )

    const gid = 'g_' + cid
    nodeGroups.push({ id: gid, label, isKnownConcept: isKg, conceptId: kgId, terms: allTerms })
    for (const item of items) itemToGroup.set(item, gid)
  }

  // 移除明显不是概念的组（章节标题、疑问句、碎片）
  const nonConcept = (s: string) =>
    /^(为什么|什么|怎么|如何|是否)/.test(s) ||
    /(驱动|理解|衍生|概念)$/.test(s) && !conceptCandidates.has(s.toLowerCase())
  const filteredGroups = nodeGroups.filter(ng =>
    ng.isKnownConcept || !nonConcept(ng.label)
  )

  // 社区间边
  const edgeMap = new Map<string, number>()
  for (const items of sentenceItems) {
    const arr = [...items]
    const gs = new Set<string>()
    for (const item of arr) { const g = itemToGroup.get(item); if (g) gs.add(g) }
    const ga = [...gs]
    for (let i = 0; i < ga.length; i++) {
      for (let j = i + 1; j < ga.length; j++) {
        const key = [ga[i], ga[j]].sort().join('--')
        edgeMap.set(key, (edgeMap.get(key) ?? 0) + 1)
      }
    }
  }

  const edges: GraphEdge[] = []
  for (const [key, w] of edgeMap) {
    const [s, t] = key.split('--')
    edges.push({ sourceId: s, targetId: t, weight: w })
  }

  return { nodeGroups: filteredGroups, edges }
}

// ========== 社区距离过滤 ==========

/**
 * BFS 计算社区间距，过滤距离核心概念过远的社区
 */
export function filterFarGroups(
  graph: TextGraph,
  coreId: string,
  allConcepts: Concept[]
): TextGraph {
  if (!coreId || graph.nodeGroups.length <= 1) return graph

  // 构建社区邻接表
  const adj = new Map<string, Set<string>>()
  for (const e of graph.edges) {
    if (!adj.has(e.sourceId)) adj.set(e.sourceId, new Set())
    if (!adj.has(e.targetId)) adj.set(e.targetId, new Set())
    adj.get(e.sourceId)!.add(e.targetId)
    adj.get(e.targetId)!.add(e.sourceId)
  }

  // 找核心社区：包含 coreId 或 coreId 对应概念名的社区
  const coreIdLower = coreId.toLowerCase()
  const coreConcept = allConcepts.find(c => c.id === coreId)
  const coreNames = new Set([coreIdLower])
  if (coreConcept) {
    coreNames.add(coreConcept.title.toLowerCase())
    if (coreConcept.alias) {
      for (const a of coreConcept.alias) coreNames.add(a.toLowerCase())
    }
  }

  let coreGroup = graph.nodeGroups[0].id
  outer: for (const ng of graph.nodeGroups) {
    for (const t of ng.terms) {
      if (coreNames.has(t.toLowerCase())) { coreGroup = ng.id; break outer }
    }
    if (coreNames.has(ng.label.toLowerCase())) { coreGroup = ng.id; break outer }
  }

  // 没有社区间边时，不过滤（所有社区视为同层级）
  if (adj.size === 0) return graph

  // BFS 算距离
  const dist = new Map<string, number>()
  const queue = [coreGroup]
  dist.set(coreGroup, 0)
  for (const cur of queue) {
    const d = dist.get(cur)!
    const neighbors = adj.get(cur)
    if (neighbors) {
      for (const n of neighbors) {
        if (!dist.has(n)) {
          dist.set(n, d + 1)
          if (d + 1 < 3) queue.push(n)
        }
      }
    }
  }

  // 过滤：只保留距离 <= 2 的社区
  const keepIds = new Set([...dist.keys()].filter(id => (dist.get(id) ?? 99) <= 2))
  if (keepIds.size === 0) keepIds.add(coreGroup)

  return {
    nodeGroups: graph.nodeGroups.filter(ng => keepIds.has(ng.id)),
    edges: graph.edges.filter(e => keepIds.has(e.sourceId) && keepIds.has(e.targetId)),
  }
}

// ========== Graph Alignment ==========

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

// ========== StructuralRank: 位置加权标签 + 社区发现 ==========

/**
 * StructuralRank: 社区发现分组 + 位置加权选标签
 *
 * 从对比实验中得到的洞察：
 * - 位置加权能准确识别真正的概念（出现在标题/冒号前/列表项）
 * - 社区发现能把相关术语分到一组
 * - 组合：社区分组 + 组内选位置加权最高的术语 = 干净的知识点
 */
export function structuralExtract(
  text: string,
  allConcepts: Concept[]
): TextGraph {
  // 1. 先用现有管线建社区
  const base = buildConceptGraphFromText(text, allConcepts)

  // 2. 算位置加权得分
  const pw = positionWeighted(text, allConcepts)
  const pwMap = new Map(pw.map(t => [t.term.toLowerCase(), t.score]))

  // 3. 每个社区：选标签（KG概念 > 位置加权高分 > 原标题）
  const nodeGroups = base.nodeGroups.map(ng => {
    // 先检查社区中是否有 KG 概念
    const kgMatch = ng.terms.map(t => {
      const c = allConcepts.find(c => c.id === t || c.title.toLowerCase() === t.toLowerCase())
      return c ? { term: t, concept: c } : null
    }).find(Boolean)
    if (kgMatch) {
      return { ...ng, label: kgMatch.concept.title, isKnownConcept: true, conceptId: kgMatch.concept.id }
    }

    // 没有 KG 概念时，用位置加权最高的术语
    let bestLabel = ng.label
    let bestScore = pwMap.get(ng.label.toLowerCase()) ?? 0
    for (const t of ng.terms) {
      const s = pwMap.get(t.toLowerCase()) ?? 0
      if (s > bestScore) { bestScore = s; bestLabel = t }
    }

    return { ...ng, label: bestLabel }
  })

  return { nodeGroups, edges: base.edges }
}

// ========== Method 2: TextRank (PageRank on term graph) ==========

export interface RankedTerm {
  term: string
  score: number
}

/**
 * TextRank: 基于 PageRank 的图排序关键词提取
 * 节点 = 候选术语 (KG概念 + 中文短语 + 英文术语)
 * 边 = 滑窗共现 (window=5)
 */
export function textrank(
  text: string,
  allConcepts: Concept[]
): RankedTerm[] {
  const terms = extractTerms(text)
  if (terms.length < 2) return []

  // 给每个术语分配序号
  const termIdx = new Map<string, number>()
  terms.forEach((t, i) => termIdx.set(t, i))
  const n = terms.length

  // 将文本切分为 term 序列（顺序出现）
  const lower = text.toLowerCase()
  const sorted = [...terms].sort((a, b) => b.length - a.length)

  // 扫描文本，按出现顺序记录术语
  const sequence: string[] = []
  let pos = 0
  while (pos < lower.length) {
    let matched = false
    for (const t of sorted) {
      if (lower.startsWith(t, pos)) {
        sequence.push(t)
        pos += t.length
        matched = true
        break
      }
    }
    if (!matched) pos++
  }

  if (sequence.length < 2) return []

  // 建共现图（滑窗 5）
  const graph = new Map<number, Set<number>>()
  for (let i = 0; i < n; i++) graph.set(i, new Set())

  for (let i = 0; i < sequence.length; i++) {
    for (let j = i + 1; j < Math.min(i + 5, sequence.length); j++) {
      if (sequence[i] !== sequence[j]) {
        const a = termIdx.get(sequence[i])!
        const b = termIdx.get(sequence[j])!
        graph.get(a)!.add(b)
        graph.get(b)!.add(a)
      }
    }
  }

  // PageRank 迭代
  const d = 0.85
  let scores = new Array(n).fill(1 / n)

  for (let iter = 0; iter < 30; iter++) {
    const newScores = new Array(n).fill(0)
    for (let i = 0; i < n; i++) {
      const neighbors = graph.get(i)!
      if (neighbors.size === 0) {
        newScores[i] = (1 - d) * scores[i]
      } else {
        let sum = 0
        for (const j of neighbors) {
          sum += scores[j] / graph.get(j)!.size
        }
        newScores[i] = (1 - d) + d * sum
      }
    }
    scores = newScores
  }

  // 返回排序结果
  return terms
    .map((term, i) => ({ term, score: scores[i] }))
    .sort((a, b) => b.score - a.score)
}

// ========== Method 3: Position-weighted Scoring ==========

/**
 * 基于文档位置的术语加权：
 * - 标题行 × 3
 * - 冒号前概念 × 2.5
 * - 列表项 × 2
 * - 正文 × 1
 */
export function positionWeighted(
  raw: string,
  allConcepts: Concept[]
): RankedTerm[] {
  const text = scrub(raw)
  const terms = extractTerms(raw)
  const scores = new Map<string, number>()

  for (const t of terms) scores.set(t, 1)

  // 标题行加权
  for (const line of text.split('\n')) {
    const clean = line.replace(/^[\s#\-*]+/, '').trim()
    if (clean.length >= 2) {
      for (const t of terms) {
        if (clean.toLowerCase().includes(t.toLowerCase()) && clean.length <= 30) {
          // 是标题行本身
          const isHeading = /^##?\s/.test(line) || /^[\s#\-*]+$/.test(line)
          if (isHeading || clean === t) {
            scores.set(t, (scores.get(t) ?? 1) + 3)
          }
          // 冒号前
          if (line.includes('：') || line.includes(':')) {
            const beforeColon = line.split(/[：:]/)[0]
            if (beforeColon.toLowerCase().includes(t.toLowerCase())) {
              scores.set(t, (scores.get(t) ?? 1) + 2.5)
            }
          }
        }
      }
    }
  }

  // 列表项加权
  for (const line of text.split('\n')) {
    if (/^[\s\-*+]/.test(line) || /^\d+[.、]/.test(line)) {
      for (const t of terms) {
        if (line.toLowerCase().includes(t.toLowerCase())) {
          scores.set(t, (scores.get(t) ?? 1) + 2)
        }
      }
    }
  }

  return [...scores.entries()]
    .map(([term, score]) => ({ term, score }))
    .sort((a, b) => b.score - a.score)
}

// ========== Method 4: Fusion Strategy ==========

/**
 * 融合策略：TextRank + 位置加权 + 社区发现
 * 1. TextRank 排序候选术语
 * 2. 位置加权调整得分
 * 3. 社区发现合并同义/相关项
 * 4. 每组选最高分术语为标签
 */
export function fusionExtract(
  text: string,
  allConcepts: Concept[]
): TextGraph {
  const tr = textrank(text, allConcepts)
  const pw = positionWeighted(text, allConcepts)

  // 融合得分：TextRank * 0.5 + 位置加权 * 0.3 + 社区拓扑 * 0.2
  const trMap = new Map(tr.map(t => [t.term, t.score]))
  const pwMap = new Map(pw.map(t => [t.term, t.score]))

  const maxTr = tr.length > 0 ? tr[0].score : 1
  const maxPw = pw.length > 0 ? pw[0].score : 1

  const allTerms = new Set([...trMap.keys(), ...pwMap.keys()])
  const fusionScores = new Map<string, number>()
  for (const t of allTerms) {
    const trScore = (trMap.get(t) ?? 0) / maxTr
    const pwScore = (pwMap.get(t) ?? 1) / maxPw
    fusionScores.set(t, trScore * 0.5 + pwScore * 0.3)
  }

  // 复用社区发现来分组
  const baseGraph = buildConceptGraphFromText(text, allConcepts)

  // 用融合得分替换社区标签选择
  const nodeGroups = baseGraph.nodeGroups.map(ng => {
    // 在社区的 terms 中找融合得分最高的作为新标签
    let bestTerm = ng.label
    let bestScore = fusionScores.get(ng.label.toLowerCase()) ?? 0
    for (const t of ng.terms) {
      const s = fusionScores.get(t.toLowerCase()) ?? 0
      if (s > bestScore) { bestScore = s; bestTerm = t }
    }
    return { ...ng, label: bestTerm }
  })

  return { nodeGroups, edges: baseGraph.edges }
}

export function compareTexts(
  userText: string,
  originalContent: string,
  allConcepts: Concept[],
  subjectConceptId?: string
): GraphAlignmentResult {
  const userGraph = buildConceptGraphFromText(userText, allConcepts)
  let originalGraph = buildConceptGraphFromText(originalContent, allConcepts)
  if (subjectConceptId) {
    originalGraph = filterFarGroups(originalGraph, subjectConceptId, allConcepts)
  }
  return alignGraphs(userGraph, originalGraph)
}
