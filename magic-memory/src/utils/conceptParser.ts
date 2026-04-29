import type { Concept, ConceptEdge } from '../types'

interface ParsedFrontmatter {
  id?: string
  title?: string
  alias?: string[]
  level?: number
  category?: string
  depends_on?: string[]
  leads_to?: string[]
  related?: string[]
}

export function parseFrontmatter(content: string): { meta: ParsedFrontmatter; body: string } {
  const fmRegex = /^---\n([\s\S]*?)\n---/
  const match = content.match(fmRegex)
  
  if (!match) {
    return { meta: {}, body: content }
  }
  
  const fmRaw = match[1]
  const body = content.slice(match[0].length).trim()
  const meta: ParsedFrontmatter = {}
  
  fmRaw.split('\n').forEach(line => {
    const colonIdx = line.indexOf(':')
    if (colonIdx === -1) return
    
    const key = line.slice(0, colonIdx).trim()
    let value = line.slice(colonIdx + 1).trim()
    
    if (value.startsWith('[') && value.endsWith(']')) {
      meta[key as keyof ParsedFrontmatter] = JSON.parse(value)
    } else if (!isNaN(Number(value))) {
      meta[key as keyof ParsedFrontmatter] = Number(value) as any
    } else {
      meta[key as keyof ParsedFrontmatter] = value as any
    }
  })
  
  return { meta, body }
}

export function parseConceptFromMarkdown(
  id: string,
  content: string,
  filePath: string
): Concept {
  const { meta, body } = parseFrontmatter(content)
  
  return {
    id: meta.id || id.replace('.md', ''),
    title: meta.title || id.replace('.md', ''),
    alias: meta.alias,
    level: meta.level || 1,
    category: meta.category || '其他',
    problem: '',
    gap_anticipate: '',
    depends_on: meta.depends_on || [],
    leads_to: meta.leads_to || [],
    related: meta.related || [],
    path: filePath,
    tags: [],
    lastModified: new Date()
  }
}

export function buildEdgesFromConcepts(concepts: Concept[]): ConceptEdge[] {
  const edges: ConceptEdge[] = []
  const conceptMap = new Map(concepts.map(c => [c.id, c]))
  
  concepts.forEach(source => {
    source.depends_on.forEach(targetId => {
      if (conceptMap.has(targetId)) {
        edges.push({
          id: `${source.id}-depends-${targetId}`,
          source: source.id,
          target: targetId,
          type: 'depends_on'
        })
      }
    })
    
    source.leads_to.forEach(targetId => {
      if (conceptMap.has(targetId)) {
        edges.push({
          id: `${source.id}-leads-${targetId}`,
          source: source.id,
          target: targetId,
          type: 'leads_to'
        })
      }
    })
    
    source.related.forEach(targetId => {
      if (conceptMap.has(targetId)) {
        edges.push({
          id: `${source.id}-related-${targetId}`,
          source: source.id,
          target: targetId,
          type: 'related'
        })
      }
    })
  })
  
  return edges
}

export function getConceptsForReview(
  concepts: Concept[],
  reviewRecords: Map<string, any>,
  limit: number = 10
): Concept[] {
  const now = new Date()
  const due: Concept[] = []
  
  concepts.forEach(concept => {
    const record = reviewRecords.get(concept.id)
    if (!record) {
      due.push(concept)
      return
    }
    
    const nextReview = new Date(record.next_review)
    if (nextReview <= now) {
      due.push(concept)
    }
  })
  
  return due.slice(0, limit)
}

export function calculateNextReview(
  easeFactor: number,
  interval: number,
  quality: number
): { interval: number; easeFactor: number } {
  let newInterval = interval
  let newEaseFactor = easeFactor
  
  if (quality < 3) {
    newInterval = 1
  } else if (interval === 0) {
    newInterval = 1
  } else if (interval === 1) {
    newInterval = 6
  } else {
    newInterval = Math.round(interval * easeFactor)
  }
  
  newEaseFactor = easeFactor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
  newEaseFactor = Math.max(1.3, newEaseFactor)
  
  return { interval: newInterval, easeFactor: newEaseFactor }
}

/**
 * 将 LLM 返回的标题列表匹配到已有概念的 ID
 * 支持精确匹配、别名匹配、模糊匹配
 */
export function matchTitlesToIds(
  titles: string[],
  concepts: { id: string; title: string; alias?: string[] }[]
): string[] {
  const ids: string[] = []
  for (const title of titles) {
    const t = title.trim()
    if (!t) continue
    // 精确匹配
    let found = concepts.find(c => c.title === t || (c.alias && c.alias.includes(t)))
    if (found) { ids.push(found.id); continue }
    // 包含匹配
    found = concepts.find(c => c.title.includes(t) || t.includes(c.title))
    if (found) { ids.push(found.id); continue }
    // 别名包含匹配
    found = concepts.find(c => c.alias && c.alias.some(a => a.includes(t) || t.includes(a)))
    if (found) { ids.push(found.id); continue }
    // 归一化匹配（去空格、特殊字符）
    const norm = t.replace(/[\s\-_]/g, '').toLowerCase()
    found = concepts.find(c =>
      c.title.replace(/[\s\-_]/g, '').toLowerCase() === norm ||
      (c.alias && c.alias.some(a => a.replace(/[\s\-_]/g, '').toLowerCase() === norm))
    )
    if (found) { ids.push(found.id); continue }
  }
  return ids
}