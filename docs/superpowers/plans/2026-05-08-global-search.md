# Global Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a global search bar to the knowledge graph that lets users find concepts by title, alias, or problem description.

**Architecture:** A new `GlobalSearch` component filters the existing `concepts[]` array from the Zustand store client-side. It renders as an absolutely-positioned search bar at the top of the graph area. Matches are scored by precision (exact > prefix > substring > alias > problem). Clicking a result calls `handleSelectConcept`, which selects the concept and focuses the graph on it.

**Tech Stack:** React 19, TypeScript, Zustand, Tailwind CSS 4, Cytoscape.js

**Files:**
- Create: `src/components/GlobalSearch.tsx`
- Modify: `src/components/KnowledgeGraphView.tsx`

---

### Task 1: Create GlobalSearch component

**Files:**
- Create: `src/components/GlobalSearch.tsx`

- [ ] **Step 1: Write the component**

Create `src/components/GlobalSearch.tsx`:

```tsx
import { useState, useRef, useEffect, useMemo } from 'react'
import type { Concept } from '../types'

interface GlobalSearchProps {
  concepts: Concept[]
  onSelect: (concept: Concept) => void
}

interface MatchResult {
  concept: Concept
  score: number
  matchField: 'title' | 'alias' | 'problem'
}

function matchScore(concept: Concept, query: string): MatchResult | null {
  const q = query.toLowerCase().trim()
  if (!q) return null

  const title = concept.title.toLowerCase()
  const aliases = concept.alias?.map(a => a.toLowerCase()) ?? []
  const problem = concept.problem?.toLowerCase() ?? ''

  if (title === q) return { concept, score: 100, matchField: 'title' }
  if (title.startsWith(q)) return { concept, score: 80, matchField: 'title' }
  if (title.includes(q)) return { concept, score: 60, matchField: 'title' }

  for (const alias of aliases) {
    if (alias.includes(q)) return { concept, score: 40, matchField: 'alias' }
  }

  if (problem.includes(q)) return { concept, score: 20, matchField: 'problem' }

  return null
}

const MAX_RESULTS = 10

export function GlobalSearch({ concepts, onSelect }: GlobalSearchProps) {
  const [query, setQuery] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const [highlightIndex, setHighlightIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const results = useMemo(() => {
    if (!query.trim()) return []
    const scored = concepts
      .map(c => matchScore(c, query))
      .filter((r): r is MatchResult => r !== null)
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_RESULTS)
    return scored
  }, [concepts, query])

  // Reset highlight when results change
  useEffect(() => {
    setHighlightIndex(0)
  }, [results.length])

  // Click outside to close
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleSelect = (concept: Concept) => {
    onSelect(concept)
    setQuery('')
    setIsOpen(false)
    inputRef.current?.blur()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlightIndex(i => Math.min(i + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlightIndex(i => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (results[highlightIndex]) {
        handleSelect(results[highlightIndex].concept)
      }
    } else if (e.key === 'Escape') {
      setIsOpen(false)
      inputRef.current?.blur()
    }
  }

  const getMatchLabel = (field: MatchResult['matchField']): string => {
    switch (field) {
      case 'title': return '标题匹配'
      case 'alias': return '别名匹配'
      case 'problem': return '问题匹配'
    }
  }

  const getLevelLabel = (level: number): string => `L${level}`
  const getLevelClasses = (level: number): string => {
    switch (level) {
      case 1: return 'bg-green-100 text-green-700'
      case 2: return 'bg-blue-100 text-blue-700'
      case 3: return 'bg-purple-100 text-purple-700'
      default: return 'bg-gray-100 text-gray-600'
    }
  }

  return (
    <div ref={containerRef} className="relative flex-1 max-w-md">
      <div className="relative">
        <svg
          className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none"
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={e => {
            setQuery(e.target.value)
            setIsOpen(true)
          }}
          onFocus={() => { if (query.trim()) setIsOpen(true) }}
          onKeyDown={handleKeyDown}
          placeholder="搜索概念..."
          className="w-full pl-9 pr-3 py-1.5 text-sm bg-gray-100 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 focus:bg-white placeholder-gray-400 transition-colors"
        />
      </div>

      {isOpen && query.trim() && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 max-h-80 overflow-y-auto">
          {results.length > 0 ? (
            <ul className="py-1">
              {results.map((result, index) => (
                <li key={result.concept.id}>
                  <button
                    className={`w-full flex items-center gap-2 px-3 py-2 text-left text-sm transition-colors ${
                      index === highlightIndex
                        ? 'bg-blue-50 text-blue-700'
                        : 'text-gray-700 hover:bg-gray-50'
                    }`}
                    onMouseDown={e => {
                      e.preventDefault()
                      handleSelect(result.concept)
                    }}
                    onMouseEnter={() => setHighlightIndex(index)}
                  >
                    <svg className="w-4 h-4 shrink-0 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="font-medium truncate">{result.concept.title}</span>
                        <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium ${getLevelClasses(result.concept.level)}`}>
                          {getLevelLabel(result.concept.level)}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        {result.concept.category && (
                          <span className="text-[10px] text-gray-400 truncate">{result.concept.category}</span>
                        )}
                        <span className="text-[10px] text-gray-300">·</span>
                        <span className="text-[10px] text-gray-400">{getMatchLabel(result.matchField)}</span>
                      </div>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="px-3 py-4 text-sm text-gray-400 text-center">
              未找到匹配的概念
            </div>
          )}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Verify no type errors**

Run: `npx tsc --noEmit --pretty 2>&1 | head -30`
Expected: No errors related to `src/components/GlobalSearch.tsx`

- [ ] **Step 3: Commit**

```bash
git add src/components/GlobalSearch.tsx
git commit -m "feat: add GlobalSearch component with real-time concept filtering"
```

---

### Task 2: Integrate GlobalSearch into KnowledgeGraphView

**Files:**
- Modify: `src/components/KnowledgeGraphView.tsx`

- [ ] **Step 1: Import the GlobalSearch component**

Add import at the top of `KnowledgeGraphView.tsx` (line ~11, after existing imports):

```tsx
import { GlobalSearch } from './GlobalSearch'
```

- [ ] **Step 2: Add search bar to the graph view**

Find the focus bar section at lines 292-301 (the `<div>` with className `absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-4 py-2 bg-white/80 backdrop-blur-sm border-b border-blue-200 shadow-sm`).

Replace the whole conditional block with a version that always shows the bar (with search) when the graph is visible (not in process mode):

Replace this block (lines 292-301):

```tsx
        {/* 聚焦模式顶部悬浮条 — 当概念被选中时显示 */}
        {selectedConcept && !processMode && (
          <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-4 py-2 bg-white/80 backdrop-blur-sm border-b border-blue-200 shadow-sm">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-sm font-medium text-gray-700">
                聚焦: <span className="text-blue-600">{selectedConcept.title}</span>
              </span>
            </div>
          </div>
        )}
```

With:

```tsx
        {/* 顶部栏：搜索 + 聚焦信息 */}
        {!processMode && (
          <div className="absolute top-0 left-0 right-0 z-20 flex items-center gap-3 px-4 py-2 bg-white/80 backdrop-blur-sm border-b border-blue-200 shadow-sm">
            <GlobalSearch
              concepts={concepts}
              onSelect={(concept) => {
                handleSelectConcept(concept)
              }}
            />
            {selectedConcept && (
              <div className="flex items-center gap-2 shrink-0">
                <span className="w-2 h-2 rounded-full bg-green-500" />
                <span className="text-sm font-medium text-gray-700">
                  聚焦: <span className="text-blue-600">{selectedConcept.title}</span>
                </span>
              </div>
            )}
          </div>
        )}
```

- [ ] **Step 3: Verify no type errors**

Run: `npx tsc --noEmit --pretty 2>&1 | head -30`
Expected: No new type errors

- [ ] **Step 4: Commit**

```bash
git add src/components/KnowledgeGraphView.tsx
git commit -m "feat: integrate GlobalSearch into knowledge graph top bar"
```

---

### Task 3: Verify complete integration

- [ ] **Step 1: Run type check**

Run: `npx tsc --noEmit --pretty`
Expected: Exit code 0, no errors related to the changed files

- [ ] **Step 2: Verify file integrity**

Run: `git diff --stat HEAD~2`
Expected: Shows 2 files changed (1 created, 1 modified)

- [ ] **Step 3: Final commit if needed**

```bash
git add -A
git commit -m "chore: finalize global search implementation" || true
```
