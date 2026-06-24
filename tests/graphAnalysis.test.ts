import { describe, it, expect } from 'vitest'
import { computeDepthLevels } from '../src/utils/graphAnalysis'
import type { Concept, ConceptEdge } from '../src/types'

describe('computeDepthLevels', () => {
  it('should return empty map for empty concepts', () => {
    const result = computeDepthLevels([], [])
    expect(result.size).toBe(0)
  })

  it('should assign depth 0 to concepts with no dependencies', () => {
    const concepts: Concept[] = [
      { id: 'a', title: 'A', level: 1, category: 'cat', depends_on: [], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'b', title: 'B', level: 1, category: 'cat', depends_on: [], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
    ]
    const result = computeDepthLevels(concepts, [])
    expect(result.get('a')).toBe(0)
    expect(result.get('b')).toBe(0)
  })

  it('should assign depth 1 to concepts that depend on depth-0 concepts', () => {
    const concepts: Concept[] = [
      { id: 'a', title: 'A', level: 1, category: 'cat', depends_on: [], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'b', title: 'B', level: 1, category: 'cat', depends_on: ['a'], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
    ]
    const edges: ConceptEdge[] = [
      { id: 'e1', source: 'b', target: 'a', type: 'depends_on' },
    ]
    const result = computeDepthLevels(concepts, edges)
    expect(result.get('a')).toBe(0)
    expect(result.get('b')).toBe(1)
  })

  it('should handle a chain A→B→C', () => {
    const concepts: Concept[] = [
      { id: 'a', title: 'A', level: 1, category: 'cat', depends_on: [], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'b', title: 'B', level: 1, category: 'cat', depends_on: ['a'], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'c', title: 'C', level: 1, category: 'cat', depends_on: ['b'], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
    ]
    const edges: ConceptEdge[] = [
      { id: 'e1', source: 'b', target: 'a', type: 'depends_on' },
      { id: 'e2', source: 'c', target: 'b', type: 'depends_on' },
    ]
    const result = computeDepthLevels(concepts, edges)
    expect(result.get('a')).toBe(0)
    expect(result.get('b')).toBe(1)
    expect(result.get('c')).toBe(2)
  })

  it('should handle diamond dependency', () => {
    const concepts: Concept[] = [
      { id: 'a', title: 'A', level: 1, category: 'cat', depends_on: [], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'b', title: 'B', level: 1, category: 'cat', depends_on: ['a'], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'c', title: 'C', level: 1, category: 'cat', depends_on: ['a'], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'd', title: 'D', level: 1, category: 'cat', depends_on: ['b', 'c'], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
    ]
    const edges: ConceptEdge[] = [
      { id: 'e1', source: 'b', target: 'a', type: 'depends_on' },
      { id: 'e2', source: 'c', target: 'a', type: 'depends_on' },
      { id: 'e3', source: 'd', target: 'b', type: 'depends_on' },
      { id: 'e4', source: 'd', target: 'c', type: 'depends_on' },
    ]
    const result = computeDepthLevels(concepts, edges)
    expect(result.get('a')).toBe(0)
    expect(result.get('b')).toBe(1)
    expect(result.get('c')).toBe(1)
    expect(result.get('d')).toBe(2)
  })

  it('should handle circular dependencies gracefully', () => {
    const concepts: Concept[] = [
      { id: 'a', title: 'A', level: 1, category: 'cat', depends_on: ['c'], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'b', title: 'B', level: 1, category: 'cat', depends_on: ['a'], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'c', title: 'C', level: 1, category: 'cat', depends_on: ['b'], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
    ]
    const edges: ConceptEdge[] = [
      { id: 'e1', source: 'a', target: 'c', type: 'depends_on' },
      { id: 'e2', source: 'b', target: 'a', type: 'depends_on' },
      { id: 'e3', source: 'c', target: 'b', type: 'depends_on' },
    ]
    const result = computeDepthLevels(concepts, edges)
    expect(result.size).toBe(3)
  })

  it('should treat leads_to as reversed depends_on', () => {
    const concepts: Concept[] = [
      { id: 'a', title: 'A', level: 1, category: 'cat', depends_on: [], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'b', title: 'B', level: 1, category: 'cat', depends_on: [], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
    ]
    const edges: ConceptEdge[] = [
      { id: 'e1', source: 'a', target: 'b', type: 'leads_to' },
    ]
    const result = computeDepthLevels(concepts, edges)
    expect(result.get('a')).toBe(0)
    expect(result.get('b')).toBe(1)
  })

  it('should ignore related edges', () => {
    const concepts: Concept[] = [
      { id: 'a', title: 'A', level: 1, category: 'cat', depends_on: [], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
      { id: 'b', title: 'B', level: 1, category: 'cat', depends_on: [], leads_to: [], related: [], path: '', tags: [], lastModified: new Date() },
    ]
    const edges: ConceptEdge[] = [
      { id: 'e1', source: 'a', target: 'b', type: 'related' },
    ]
    const result = computeDepthLevels(concepts, edges)
    expect(result.get('a')).toBe(0)
    expect(result.get('b')).toBe(0)
  })
})
