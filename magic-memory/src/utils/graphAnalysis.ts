import type { Concept, ConceptEdge } from '../types'

// ========== 输出类型 ==========

export interface RootConceptInfo {
  id: string
  title: string
  level: number
  category: string
  inDegree: number
  outDegree: number
}

export interface FlowPath {
  name: string
  pathIds: string[]
  pathTitles: string[]
  length: number
  description?: string
}

export interface HubConceptInfo {
  id: string
  title: string
  level: number
  category: string
  inDegree: number
  outDegree: number
  totalDegree: number
  dependsOnDegree: number
  leadsToDegree: number
  relatedDegree: number
}

export interface CrossLayerJump {
  source: { id: string; title: string; level: number }
  target: { id: string; title: string; level: number }
  edgeType: string
  levelGap: number
}

export interface LevelGroupInfo {
  level: number
  count: number
  concepts: { id: string; title: string; category: string }[]
  categoryGroups: Record<string, { id: string; title: string }[]>
}

export interface GraphStats {
  totalConcepts: number
  totalEdges: number
  rootsCount: number
  leavesCount: number
  crossLayerJumpsCount: number
  averageDegree: number
  density: number
  byLevel: Record<number, number>
  byCategory: Record<string, number>
  byEdgeType: Record<string, number>
}

export interface GraphAnalysis {
  /** 根节点：入度为 0 的概念 */
  rootConcepts: RootConceptInfo[]
  /** 叶子节点：出度为 0 的概念 */
  leafConcepts: RootConceptInfo[]
  /** 最长路径（全图统一分析，不分边类型） */
  longestPaths: FlowPath[]
  /** 枢纽节点：按总连接数排序 */
  hubConcepts: HubConceptInfo[]
  /** 跨层跳跃：连接不同 level 的边 */
  crossLayerJumps: CrossLayerJump[]
  /** 层级分组 */
  levelGroups: LevelGroupInfo[]
  /** 图谱统计 */
  stats: GraphStats
}

// ========== 内部辅助类型 ==========

interface AdjacencyInfo {
  inbound: { id: string; type: string }[]
  outbound: { id: string; type: string }[]
}

// ========== 核心分析函数 ==========

/**
 * 对概念图谱进行全面分析，自动推导：
 * - 根节点/叶节点
 * - 数据流路径和依赖链
 * - 枢纽节点
 * - 跨层跳跃
 * - 层级分组和统计
 */
export function analyzeGraph(concepts: Concept[], edges: ConceptEdge[]): GraphAnalysis {
  const conceptMap = new Map(concepts.map(c => [c.id, c]))
  const adj = buildAdjacency(concepts, edges)

  const rootConcepts = findRootConcepts(concepts, adj)
  const leafConcepts = findLeafConcepts(concepts, adj)
  const longestPaths = findLongestPaths(concepts, edges, conceptMap)
  const hubConcepts = findHubs(concepts, adj)
  const crossLayerJumps = findCrossLayerJumps(edges, conceptMap)
  const levelGroups = groupByLevel(concepts)
  const stats = computeStats(concepts, edges, adj)

  return {
    rootConcepts,
    leafConcepts,
    longestPaths,
    hubConcepts,
    crossLayerJumps,
    levelGroups,
    stats,
  }
}

// ========== 1. 邻接表构建 ==========

function buildAdjacency(
  concepts: Concept[],
  edges: ConceptEdge[],
): Map<string, AdjacencyInfo> {
  const adj = new Map<string, AdjacencyInfo>()

  for (const c of concepts) {
    adj.set(c.id, { inbound: [], outbound: [] })
  }

  for (const e of edges) {
    const src = adj.get(e.source)
    const tgt = adj.get(e.target)
    if (src) src.outbound.push({ id: e.target, type: e.type })
    if (tgt) tgt.inbound.push({ id: e.source, type: e.type })
  }

  return adj
}

// ========== 2. 根节点 / 叶节点 ==========

function findRootConcepts(
  concepts: Concept[],
  adj: Map<string, AdjacencyInfo>,
): RootConceptInfo[] {
  const result: RootConceptInfo[] = []

  for (const c of concepts) {
    const info = adj.get(c.id)
    if (!info) continue

    const inDegree = info.inbound.length
    const outDegree = info.outbound.length

    if (inDegree === 0) {
      result.push({ id: c.id, title: c.title, level: c.level, category: c.category, inDegree, outDegree })
    }
  }

  return result.sort((a, b) => a.level - b.level || a.title.localeCompare(b.title))
}

function findLeafConcepts(
  concepts: Concept[],
  adj: Map<string, AdjacencyInfo>,
): RootConceptInfo[] {
  const result: RootConceptInfo[] = []

  for (const c of concepts) {
    const info = adj.get(c.id)
    if (!info) continue

    const inDegree = info.inbound.length
    const outDegree = info.outbound.length

    if (outDegree === 0) {
      result.push({ id: c.id, title: c.title, level: c.level, category: c.category, inDegree, outDegree })
    }
  }

  return result.sort((a, b) => b.level - a.level || a.title.localeCompare(b.title))
}

// ========== 3. 最长路径（全图统一分析，不分边类型） ==========

function findLongestPaths(
  concepts: Concept[],
  edges: ConceptEdge[],
  conceptMap: Map<string, Concept>,
): FlowPath[] {
  // 构建全图邻接表（所有边类型，过滤自环）
  const adj = new Map<string, string[]>()

  for (const c of concepts) {
    adj.set(c.id, [])
  }

  for (const e of edges) {
    if (e.source === e.target) continue
    const list = adj.get(e.source)
    if (list && !list.includes(e.target)) list.push(e.target)
  }

  // 从每个节点 DFS 找最长简单路径
  const allPaths: string[][] = []

  function dfs(node: string, path: string[], visited: Set<string>) {
    const neighbors = adj.get(node) ?? []
    const unvisited = neighbors.filter(n => !visited.has(n))

    if (unvisited.length === 0) {
      if (path.length >= 2) allPaths.push([...path])
      return
    }

    for (const next of unvisited) {
      visited.add(next)
      dfs(next, [...path, next], visited)
      visited.delete(next)
    }
  }

  for (const c of concepts) {
    const neighbors = adj.get(c.id)
    if (!neighbors || neighbors.length === 0) continue
    const visited = new Set<string>([c.id])
    dfs(c.id, [c.id], visited)
  }

  // 去重
  const seen = new Set<string>()
  const uniquePaths: string[][] = []
  for (const p of allPaths) {
    const key = p.join('→')
    if (!seen.has(key)) {
      seen.add(key)
      uniquePaths.push(p)
    }
  }

  uniquePaths.sort((a, b) => b.length - a.length)
  const topPaths = uniquePaths.slice(0, 5)

  return topPaths.map((path, i) => ({
    name: `路径 ${i + 1}`,
    pathIds: path,
    pathTitles: path.map(id => conceptMap.get(id)?.title ?? id).filter(Boolean),
    length: path.length,
  }))
}

// ========== 4. 依赖链 (沿 depends_on) ==========

function findDependencyChains(
  concepts: Concept[],
  edges: ConceptEdge[],
  conceptMap: Map<string, Concept>,
): FlowPath[] {
  // 构建 depends_on 邻接表 (source → target 表示 source depends on target)
  const depAdj = new Map<string, string[]>()

  for (const c of concepts) {
    depAdj.set(c.id, [])
  }

  for (const e of edges) {
    if (e.type !== 'depends_on' || e.source === e.target) continue
    const list = depAdj.get(e.source)
    if (list && !list.includes(e.target)) list.push(e.target)
  }

  // 从每个节点 DFS 找最长简单路径
  const allChains: string[][] = []

  function dfs(node: string, path: string[], visited: Set<string>) {
    const neighbors = depAdj.get(node) ?? []
    const unvisited = neighbors.filter(n => !visited.has(n))

    if (unvisited.length === 0) {
      if (path.length >= 2) allChains.push([...path])
      return
    }

    for (const next of unvisited) {
      visited.add(next)
      dfs(next, [...path, next], visited)
      visited.delete(next)
    }
  }

  for (const c of concepts) {
    const adj = depAdj.get(c.id)
    if (!adj || adj.length === 0) continue
    const visited = new Set<string>([c.id])
    dfs(c.id, [c.id], visited)
  }

  // 去重
  const seen = new Set<string>()
  const uniqueChains: string[][] = []
  for (const p of allChains) {
    const key = p.join('→')
    if (!seen.has(key)) {
      seen.add(key)
      uniqueChains.push(p)
    }
  }

  uniqueChains.sort((a, b) => b.length - a.length)

  return uniqueChains.slice(0, 5).map((chain, i) => ({
    name: `依赖链 ${i + 1}`,
    pathIds: chain,
    pathTitles: chain.map(id => conceptMap.get(id)?.title ?? id).filter(Boolean),
    length: chain.length,
  }))
}

// ========== 5. 枢纽节点 ==========

function findHubs(
  concepts: Concept[],
  adj: Map<string, AdjacencyInfo>,
): HubConceptInfo[] {
  const result: HubConceptInfo[] = []

  for (const c of concepts) {
    const info = adj.get(c.id)
    if (!info) continue

    const inDegree = info.inbound.length
    const outDegree = info.outbound.length
    const totalDegree = inDegree + outDegree

    const dependsOnDegree = info.inbound.filter(e => e.type === 'depends_on').length +
      info.outbound.filter(e => e.type === 'depends_on').length
    const leadsToDegree = info.inbound.filter(e => e.type === 'leads_to').length +
      info.outbound.filter(e => e.type === 'leads_to').length
    const relatedDegree = info.inbound.filter(e => e.type === 'related').length +
      info.outbound.filter(e => e.type === 'related').length

    result.push({
      id: c.id,
      title: c.title,
      level: c.level,
      category: c.category,
      inDegree,
      outDegree,
      totalDegree,
      dependsOnDegree,
      leadsToDegree,
      relatedDegree,
    })
  }

  return result.sort((a, b) => b.totalDegree - a.totalDegree)
}

// ========== 6. 跨层跳跃 ==========

function findCrossLayerJumps(
  edges: ConceptEdge[],
  conceptMap: Map<string, Concept>,
): CrossLayerJump[] {
  const jumps: CrossLayerJump[] = []

  for (const e of edges) {
    const src = conceptMap.get(e.source)
    const tgt = conceptMap.get(e.target)
    if (!src || !tgt) continue
    if (src.level === tgt.level) continue

    jumps.push({
      source: { id: src.id, title: src.title, level: src.level },
      target: { id: tgt.id, title: tgt.title, level: tgt.level },
      edgeType: e.type,
      levelGap: Math.abs(src.level - tgt.level),
    })
  }

  return jumps.sort((a, b) => b.levelGap - a.levelGap)
}

// ========== 7. 层级分组 ==========

function groupByLevel(concepts: Concept[]): LevelGroupInfo[] {
  const byLevel = new Map<number, Concept[]>()

  for (const c of concepts) {
    const list = byLevel.get(c.level) ?? []
    list.push(c)
    byLevel.set(c.level, list)
  }

  const result: LevelGroupInfo[] = []

  for (const [level, cons] of [...byLevel.entries()].sort(([a], [b]) => a - b)) {
    const categoryGroups: Record<string, { id: string; title: string }[]> = {}
    for (const c of cons) {
      const list = categoryGroups[c.category] ?? []
      list.push({ id: c.id, title: c.title })
      categoryGroups[c.category] = list
    }

    result.push({
      level,
      count: cons.length,
      concepts: cons.map(c => ({ id: c.id, title: c.title, category: c.category })),
      categoryGroups,
    })
  }

  return result
}

// ========== 8. 统计 ==========

function computeStats(
  concepts: Concept[],
  edges: ConceptEdge[],
  adj: Map<string, AdjacencyInfo>,
): GraphStats {
  const totalConcepts = concepts.length
  const totalEdges = edges.length

  let rootsCount = 0
  let leavesCount = 0
  let totalDegree = 0

  for (const c of concepts) {
    const info = adj.get(c.id)
    if (!info) continue
    const deg = info.inbound.length + info.outbound.length
    totalDegree += deg
    if (info.inbound.length === 0) rootsCount++
    if (info.outbound.length === 0) leavesCount++
  }

  // 按 level 统计
  const byLevel: Record<number, number> = {}
  for (const c of concepts) {
    byLevel[c.level] = (byLevel[c.level] ?? 0) + 1
  }

  // 按 category 统计
  const byCategory: Record<string, number> = {}
  for (const c of concepts) {
    byCategory[c.category] = (byCategory[c.category] ?? 0) + 1
  }

  // 按 edgeType 统计
  const byEdgeType: Record<string, number> = {}
  for (const e of edges) {
    byEdgeType[e.type] = (byEdgeType[e.type] ?? 0) + 1
  }

  // 跨层跳跃
  const crossLayerJumpsCount = edges.filter(e => {
    const src = concepts.find(c => c.id === e.source)
    const tgt = concepts.find(c => c.id === e.target)
    return src && tgt && src.level !== tgt.level
  }).length

  const maxPossibleEdges = totalConcepts * (totalConcepts - 1)
  const density = maxPossibleEdges > 0 ? totalEdges / maxPossibleEdges : 0

  return {
    totalConcepts,
    totalEdges,
    rootsCount,
    leavesCount,
    crossLayerJumpsCount,
    averageDegree: totalConcepts > 0 ? +(totalDegree / totalConcepts).toFixed(2) : 0,
    density: +density.toFixed(4),
    byLevel,
    byCategory,
    byEdgeType,
  }
}

// ========== 9. 文本格式输出 ==========

/**
 * 将分析结果格式化为可读的文本报告
 */
export function formatAnalysisToString(analysis: GraphAnalysis): string {
  const lines: string[] = []

  // 统计概览
  lines.push('## 图谱统计')
  lines.push(`| 指标 | 值 |`)
  lines.push(`|------|-----|`)
  lines.push(`| 总概念数 | ${analysis.stats.totalConcepts} |`)
  lines.push(`| 总边数 | ${analysis.stats.totalEdges} |`)
  lines.push(`| 根节点 | ${analysis.stats.rootsCount} |`)
  lines.push(`| 叶子节点 | ${analysis.stats.leavesCount} |`)
  lines.push(`| 平均度 | ${analysis.stats.averageDegree} |`)
  lines.push(`| 密度 | ${analysis.stats.density} |`)
  lines.push(`| 跨层跳跃 | ${analysis.stats.crossLayerJumpsCount} |`)
  lines.push('')
  lines.push(`**边类型分布：** ${JSON.stringify(analysis.stats.byEdgeType)}`)
  lines.push(`**层级分布：** ${JSON.stringify(analysis.stats.byLevel)}`)
  lines.push(`**分类分布：** ${JSON.stringify(analysis.stats.byCategory)}`)
  lines.push('')

  // 根节点
  lines.push('## 根节点（入度为 0）')
  lines.push('')
  if (analysis.rootConcepts.length === 0) {
    lines.push('无根节点')
  } else {
    for (const r of analysis.rootConcepts) {
      lines.push(`- **${r.title}** (L${r.level}, ${r.category}) 出度: ${r.outDegree}`)
    }
  }
  lines.push('')

  // 叶子节点
  lines.push('## 叶子节点（出度为 0）')
  lines.push('')
  if (analysis.leafConcepts.length === 0) {
    lines.push('无叶子节点')
  } else {
    for (const l of analysis.leafConcepts) {
      lines.push(`- **${l.title}** (L${l.level}, ${l.category}) 入度: ${l.inDegree}`)
    }
  }
  lines.push('')

  // 数据流路径
  lines.push('## 数据流路径（沿 leads_to 最长路径）')
  lines.push('')
  if (analysis.dataFlowPaths.length === 0) {
    lines.push('未发现数据流路径')
  } else {
    for (const p of analysis.dataFlowPaths) {
      lines.push(`### ${p.name} (${p.length} 步)`)
      lines.push('')
      lines.push(`\`\`\`\n${p.pathTitles.join(' → ')}\n\`\`\``)
      lines.push('')
    }
  }

  // 依赖链
  lines.push('## 依赖链（沿 depends_on 最长路径）')
  lines.push('')
  if (analysis.dependencyChains.length === 0) {
    lines.push('未发现依赖链')
  } else {
    for (const c of analysis.dependencyChains) {
      lines.push(`### ${c.name} (${c.length} 步)`)
      lines.push('')
      lines.push(`\`\`\`\n${c.pathTitles.join(' → ')}\n\`\`\``)
      lines.push('')
    }
  }

  // 枢纽节点
  lines.push('## 枢纽节点（按连接数排序）')
  lines.push('')
  if (analysis.hubConcepts.length === 0) {
    lines.push('无枢纽节点')
  } else {
    lines.push('| 排名 | 概念 | Level | 入度 | 出度 | 总计 | depends_on | leads_to | related |')
    lines.push('|------|------|-------|------|------|------|------------|----------|---------|')
    analysis.hubConcepts.slice(0, 15).forEach((h, i) => {
      lines.push(`| ${i + 1} | ${h.title} | L${h.level} | ${h.inDegree} | ${h.outDegree} | ${h.totalDegree} | ${h.dependsOnDegree} | ${h.leadsToDegree} | ${h.relatedDegree} |`)
    })
  }
  lines.push('')

  // 跨层跳跃
  if (analysis.crossLayerJumps.length > 0) {
    lines.push('## 跨层跳跃')
    lines.push('')
    for (const j of analysis.crossLayerJumps.slice(0, 10)) {
      const arrow = j.edgeType === 'leads_to' ? '→' : j.edgeType === 'depends_on' ? '→' : '↔'
      lines.push(`- **${j.source.title}**(L${j.source.level}) ${arrow} **${j.target.title}**(L${j.target.level}) [gap=${j.levelGap}, ${j.edgeType}]`)
    }
    lines.push('')
  }

  // 层级分组
  lines.push('## 层级分组')
  lines.push('')
  for (const g of analysis.levelGroups) {
    lines.push(`### Level ${g.level} (${g.count} 个概念)`)
    lines.push('')
    for (const [cat, cons] of Object.entries(g.categoryGroups)) {
      const titles = cons.map(c => c.title).join(', ')
      lines.push(`- **${cat}**: ${titles}`)
    }
    lines.push('')
  }

  return lines.join('\n')
}
