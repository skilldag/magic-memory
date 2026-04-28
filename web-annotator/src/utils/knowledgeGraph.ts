import type { Concept, ConceptEdge, ReviewRecord } from '../types'

export function getRelatedConcepts(conceptId: string, edges: ConceptEdge[], concepts: Concept[]) {
  return edges
    .filter(e => e.source === conceptId || e.target === conceptId)
    .map(e => {
      const otherId = e.source === conceptId ? e.target : e.source
      const concept = concepts.find(c => c.id === otherId)
      return concept ? { concept, edgeType: e.type } : null
    })
    .filter(Boolean) as { concept: Concept; edgeType: string }[]
}

export function getDependencyChain(conceptId: string, concepts: Concept[]) {
  const chain: Concept[] = []
  let current = concepts.find(c => c.id === conceptId)
  while (current && current.depends_on.length > 0) {
    const parentId = current.depends_on[0]
    const parent = concepts.find(c => c.id === parentId)
    if (parent) {
      chain.unshift(parent)
      current = parent
    } else break
  }
  return chain
}

export function getWhatIsSummary(concept: Concept) {
  const plain = concept.content
    .replace(/```[\s\S]*?```/g, '')
    .replace(/^#+\s+/gm, '')
    .replace(/[*_`>#-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  if (!plain) return ''
  return plain.length > 120 ? `${plain.slice(0, 120)}...` : plain
}

export function getRelationReason(current: Concept, other: Concept, edgeType: string) {
  if (edgeType === 'depends_on') {
    if (other.problem) return `先理解它：${other.problem}`
    return `它为「${current.title}」提供前置认知基础。`
  }
  if (edgeType === 'leads_to') {
    if (other.problem) return `学完当前概念后，下一步常会遇到：${other.problem}`
    return `它是「${current.title}」的自然延伸方向。`
  }
  if (current.category !== other.category) {
    return `它从「${other.category}」视角补充了当前概念。`
  }
  return '它与当前概念在同一主题中互补，可并行理解。'
}

export function getReviewRecordFor(conceptId: string, records: Map<string, ReviewRecord>) {
  const record = records.get(conceptId)
  if (!record) return null
  return {
    ...record,
    last_reviewed: new Date(record.last_reviewed),
    next_review: new Date(record.next_review),
  }
}
