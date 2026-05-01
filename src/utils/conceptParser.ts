import type { Concept } from '../types'

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
