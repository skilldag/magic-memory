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
  const parts: string[] = []

  if (concept.problem) {
    parts.push(concept.problem)
  }
  if (concept.gap_anticipate) {
    parts.push(concept.gap_anticipate)
  }
  if (concept.elements && concept.elements.length > 0) {
    parts.push(concept.elements.map(e => e.description).join('；'))
  }

  const summary = parts.join(' ').trim()
  if (!summary) return ''
  return summary.length > 120 ? `${summary.slice(0, 120)}...` : summary
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

export type ReviewBadgeType = {
  text: string       // "🔥" | "今日" | "New" | "✓" | "Xd"
  color: string      // Badge background color hex
  urgency: number    // 0=overdue, 1=today, 2=1-7d, 4=mastered, 5=none, 6=hidden
}

export function getReviewBadge(record: ReviewRecord | undefined): ReviewBadgeType {
  if (!record) {
    return { text: 'New', color: '#6b7280', urgency: 5 }
  }
  const now = Date.now()
  const nextReview = new Date(record.next_review).getTime()
  const diffDays = Math.ceil((nextReview - now) / (1000 * 60 * 60 * 24))
  const lastReviewed = new Date(record.last_reviewed).getTime()
  const hoursSinceReview = (now - lastReviewed) / (1000 * 60 * 60)

  // Just reviewed (< 1h ago) → no badge
  if (hoursSinceReview < 1) {
    return { text: '', color: 'transparent', urgency: 6 }
  }
  // Overdue
  if (diffDays <= 0) {
    return { text: '🔥', color: '#ef4444', urgency: 0 }
  }
  // Due today
  if (diffDays <= 1) {
    return { text: '今日', color: '#f59e0b', urgency: 1 }
  }
  // Due within 7 days
  if (diffDays <= 7) {
    return { text: `${diffDays}d`, color: '#3b82f6', urgency: 2 }
  }
  // Mastered (interval > 21)
  if (record.interval > 21) {
    return { text: '✓', color: '#10b981', urgency: 4 }
  }
  // Well-ahead, no badge
  return { text: '', color: 'transparent', urgency: 6 }
}

export function getDueConcepts(
  concepts: Concept[],
  records: Map<string, ReviewRecord>
): { concept: Concept; badge: ReviewBadgeType; daysUntilReview: number }[] {
  const now = Date.now()
  const result: { concept: Concept; badge: ReviewBadgeType; daysUntilReview: number }[] = []
  for (const c of concepts) {
    const r = records.get(c.id)
    if (!r) continue
    const badge = getReviewBadge(r)
    if (badge.urgency <= 2) {
      const diff = Math.ceil((new Date(r.next_review).getTime() - now) / (1000 * 60 * 60 * 24))
      result.push({ concept: c, badge, daysUntilReview: diff })
    }
  }
  return result.sort((a, b) => a.badge.urgency - b.badge.urgency || a.daysUntilReview - b.daysUntilReview)
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
