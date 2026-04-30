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

function extractTerms(text: string): string[] {
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
  const terms = extractTerms(text, 1)

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

  return { nodeGroups, edges }
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

export function compareTexts(
  userText: string,
  originalContent: string,
  allConcepts: Concept[]
): GraphAlignmentResult {
  return alignGraphs(
    buildConceptGraphFromText(userText, allConcepts),
    buildConceptGraphFromText(originalContent, allConcepts)
  )
}
