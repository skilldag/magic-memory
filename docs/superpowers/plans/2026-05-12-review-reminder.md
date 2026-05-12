# 记忆曲线复习提醒 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add visual review reminders (badges + banner + auto-align flow) to the knowledge graph so users can see which concepts need review and act on them with one click.

**Architecture:** Extend existing `ReviewRecord`/SM-2 infrastructure with utility functions for badge state computation. Modify KnowledgeGraph to render badges as Cytoscape overlays. Add a review banner in KnowledgeGraphView and a review queue section in SummaryPanel. Wire AlignmentPanel's alignment-complete event to call `startReview()`.

**Tech Stack:** React 19 + TypeScript + Zustand + Cytoscape + Tailwind CSS

---

### Task 1: Add badge utility functions

**Files:**
- Modify: `src/utils/knowledgeGraph.ts`

- [ ] **Step 1: Add `getReviewBadge()` and `getDueConcepts()`**

Append at end of `src/utils/knowledgeGraph.ts`:

```typescript
export type ReviewBadgeType = {
  text: string       // "🔥" | "今日" | "New" | "✓" | "Xd"
  color: string      // Badge background color hex
  urgency: number    // 0=overdue, 1=today, 2=1-3d, 3=4-7d, 4=mastered, 5=none
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
    if (!r) continue // skip un-reviewed concepts
    const badge = getReviewBadge(r)
    if (badge.urgency <= 2) { // overdue, today, or within 7 days
      const diff = Math.ceil((new Date(r.next_review).getTime() - now) / (1000 * 60 * 60 * 24))
      result.push({ concept: c, badge, daysUntilReview: diff })
    }
  }
  return result.sort((a, b) => a.badge.urgency - b.badge.urgency || a.daysUntilReview - b.daysUntilReview)
}
```

- [ ] **Step 2: Run typecheck to verify**

Run: `npx tsc --noEmit`
Expected: No type errors (the new functions use existing types `ReviewRecord` and `Concept`)

- [ ] **Step 3: Commit**

```bash
git add src/utils/knowledgeGraph.ts
git commit -m "feat: add getReviewBadge and getDueConcepts utility functions"
```

---

### Task 2: Render review badges on Cytoscape nodes

**Files:**
- Modify: `src/components/KnowledgeGraph.tsx`

- [ ] **Step 1: Add `reviewRecords` prop to KnowledgeGraph**

Import `ReviewRecord` type:
```typescript
import type { Concept, ConceptEdge, ReviewRecord } from '../types'
```

Import `getReviewBadge`:
```typescript
import { getReviewBadge } from '../utils/knowledgeGraph'
```

Add to `KnowledgeGraphProps` interface:
```typescript
  reviewRecords?: Map<string, ReviewRecord>
```

Add to destructured props in function signature:
```typescript
  reviewRecords,
```

Add ref for reviewRecords:
```typescript
  const reviewRecordsRef = useRef(reviewRecords)
  reviewRecordsRef.current = reviewRecords
```

- [ ] **Step 2: Add badge data to node elements on initialization**

In the elements construction (around line 196-205), add `badge` to node data:
```typescript
      ...concepts.map(c => ({
        group: 'nodes' as const,
        data: {
          id: c.id,
          label: c.title,
          level: c.level,
          category: c.category,
          mastery: conceptMastery?.get(c.id)?.score,
          badge: getReviewBadge(reviewRecords?.get(c.id)).text,
          badgeColor: getReviewBadge(reviewRecords?.get(c.id)).color,
        }
      })),
```

In the incremental update section (around line 479-491), add badge data:
```typescript
      if (!currentIds.has(c.id)) {
        cy.add({
          group: 'nodes',
          data: {
            id: c.id,
            label: c.title,
            level: c.level,
            category: c.category,
            mastery: conceptMasteryRef.current?.get(c.id)?.score,
            badge: getReviewBadge(reviewRecordsRef.current?.get(c.id)).text,
            badgeColor: getReviewBadge(reviewRecordsRef.current?.get(c.id)).color,
          }
        })
      }
```

- [ ] **Step 3: Add badge rendering effect**

Add a new useEffect after the linkMode effect (after line ~698):

```typescript
  // Update badges on the graph when reviewRecords change
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.nodes().forEach(n => {
      const id = n.id()
      const badge = getReviewBadge(reviewRecords?.get(id))
      n.data('badge', badge.text)
      n.data('badgeColor', badge.color)
    })
  }, [reviewRecords])
```

- [ ] **Step 4: Add badge overlay HTML**

In the JSX return, after the main cytoscape container div, add the badge overlay. We need to render badge elements on top of each node. The cleanest approach is to use an HTML overlay that tracks node positions.

Add a state to track badge positions:
```typescript
  const [badgePositions, setBadgePositions] = useState<Map<string, { x: number; y: number }>>(new Map())
```

Add a function to update positions on zoom/pan:
```typescript
  const updateBadgePositions = useCallback(() => {
    const cy = cyRef.current
    if (!cy || !reviewRecords) return
    const positions = new Map<string, { x: number; y: number }>()
    cy.nodes().forEach(n => {
      const id = n.id()
      const badge = getReviewBadge(reviewRecords.get(id))
      if (!badge.text) return // skip empty badges
      const pos = n.renderedPosition()
      const bb = n.renderedBoundingBox()
      positions.set(id, {
        x: pos.x + bb.w / 2 - 12,
        y: pos.y - bb.h / 2 - 2,
      })
    })
    setBadgePositions(positions)
  }, [reviewRecords])
```

Update positions on zoom/pan/drag:
```typescript
  // In the cy initialization, after zoom handler:
  cy.on('zoom', () => {
    setZoomLevel(cy.zoom())
    updateBadgePositions() // added
  })
  
  cy.on('pan', () => {
    updateBadgePositions() // added
  })
  
  cy.on('dragfree', 'node', () => {
    updateBadgePositions() // added
  })
```

Call `updateBadgePositions` after layout:

In the fastLayout `layoutstop` handler:
```typescript
  fastLayout.one('layoutstop', () => {
    cy.fit(undefined, 50)
    setZoomLevel(cy.zoom())
    setTimeout(() => updateBadgePositions(), 100) // wait for render
  })
```

Also call it in the `setIsReady(true)` path:
```typescript
  setIsReady(true)
  initialLayoutDoneRef.current = true
  setTimeout(() => updateBadgePositions(), 200)
```

Render the badges in JSX, after the legend div:
```typescript
  {isReady && badgePositions.size > 0 && (
    <div className="absolute inset-0 pointer-events-none" style={{ zIndex: 10 }}>
      {Array.from(badgePositions.entries()).map(([id, pos]) => {
        const badge = getReviewBadge(reviewRecords?.get(id))
        if (!badge.text) return null
        return (
          <div
            key={id}
            className="absolute flex items-center justify-center rounded-full text-[9px] font-bold text-white leading-none pointer-events-none"
            style={{
              left: pos.x,
              top: pos.y,
              width: 22,
              height: 22,
              backgroundColor: badge.color,
              boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
              transform: 'translate(-50%, -50%)',
              zIndex: 20,
            }}
          >
            {badge.text}
          </div>
        )
      })}
    </div>
  )}
```

- [ ] **Step 5: Typecheck**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add src/components/KnowledgeGraph.tsx
git commit -m "feat: render review badges on knowledge graph nodes"
```

---

### Task 3: Add review banner to KnowledgeGraphView

**Files:**
- Modify: `src/components/KnowledgeGraphView.tsx`

- [ ] **Step 1: Add banner state and computation**

Import `getDueConcepts`:
```typescript
import { getDueConcepts } from '../utils/knowledgeGraph'
```

Add state in the component:
```typescript
  const [bannerDismissed, setBannerDismissed] = useState(false)
  const reviewRecords = useKnowledgeGraphStore(s => s.reviewRecords)
```

Add computed due list:
```typescript
  const dueConcepts = useMemo(
    () => getDueConcepts(concepts, reviewRecords).filter(d => d.badge.urgency <= 1),
    [concepts, reviewRecords]
  )
```

- [ ] **Step 2: Add banner JSX**

After the search bar div (around line 315), add:

```typescript
  {!processMode && dueConcepts.length > 0 && !bannerDismissed && (
    <div className="shrink-0 flex items-center gap-3 px-4 py-2 bg-amber-50 border-b border-amber-200">
      <span className="text-sm">📅</span>
      <span className="text-sm text-amber-800">
        <strong>{dueConcepts.length}</strong> 个概念需要复习
        {dueConcepts[0] && ` · 最长的已过期 ${Math.abs(dueConcepts[0].daysUntilReview)} 天`}
      </span>
      <button
        onClick={() => {
          const first = dueConcepts[0]
          if (first) {
            const concept = concepts.find(c => c.id === first.concept.id)
            if (concept) handleSelectConcept(concept)
          }
        }}
        className="ml-auto px-3 py-1 text-xs font-medium text-amber-800 bg-amber-200/50 rounded-md hover:bg-amber-300/50 transition-colors"
      >
        查看待复习 →
      </button>
      <button
        onClick={() => setBannerDismissed(true)}
        className="w-5 h-5 flex items-center justify-center text-amber-400 hover:text-amber-600 transition-colors"
        title="关闭"
      >
        <svg width={12} height={12} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  )}
```

- [ ] **Step 3: Pass reviewRecords to KnowledgeGraph**

Find the `<KnowledgeGraph` usage (around line 401) and add the prop:
```typescript
            reviewRecords={reviewRecords}
```

- [ ] **Step 4: Trigger banner re-evaluation after concept panel alignment**

When the user completes an alignment and SM-2 updates, `reviewRecords` will change, and `dueConcepts` will auto-recompute via `useMemo`. This works naturally.

- [ ] **Step 5: Typecheck**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add src/components/KnowledgeGraphView.tsx
git commit -m "feat: add review reminder banner to knowledge graph view"
```

---

### Task 4: Auto-switch ConceptDetailPanel tab based on review badge

**Files:**
- Modify: `src/components/ConceptDetailPanel.tsx`

- [ ] **Step 1: Import review utilities**

```typescript
import { getReviewBadge } from '../utils/knowledgeGraph'
```

- [ ] **Step 2: Compute default action based on badge**

Add after the `const [action, setAction] = useState<ActionKey>('read')` line:

```typescript
  // Auto-switch tab based on review badge
  const reviewBadge = getReviewBadge(reviewRecords.get(concept.id))
  const defaultAction: ActionKey = 
    reviewBadge.text === '🔥' || reviewBadge.text === '今日'
      ? 'align'
      : 'read'

  // Use defaultAction as initial action when concept changes
  const [currentAction, setCurrentAction] = useState<ActionKey>(defaultAction)

  useEffect(() => {
    const badge = getReviewBadge(reviewRecords.get(concept.id))
    if (badge.text === '🔥' || badge.text === '今日') {
      setCurrentAction('align')
    } else {
      // Keep current action if it was already set to something other than align
      // Don't override user's manual tab choice
    }
  }, [concept.id])

  // Override the setAction to update currentAction
  const handleSetAction = useCallback((a: ActionKey) => {
    setCurrentAction(a)
  }, [])
```

Wait, this is getting complex. Let me simplify. The simplest change: when concept changes, if the concept is due for review, default to 'align'. Otherwise default to 'read'.

Actually, looking at the existing code more carefully:

```typescript
const [action, setAction] = useState<ActionKey>('read')
```

The simplest correct approach: use a key on the component or a useEffect that resets the action when concept.id changes.

Let me simplify the approach:

```typescript
  // Auto-switch to align tab for concepts due for review
  useEffect(() => {
    const badge = getReviewBadge(reviewRecords.get(concept.id))
    if (badge.text === '🔥' || badge.text === '今日') {
      setAction('align')
    } else {
      setAction('read')
    }
  }, [concept.id])
```

Add after the docContent loading useEffect.

But wait, this might conflict with the user manually switching tabs. Let me only auto-switch when the concept ID changes, not on every re-render. Since `concept.id` only changes when selecting a new concept, this won't interfere with manual tab switches.

- [ ] **Step 3: Code change**

```typescript
  // Auto-switch to align tab for concepts due for review
  useEffect(() => {
    const badge = getReviewBadge(reviewRecords.get(concept.id))
    setAction(badge.text === '🔥' || badge.text === '今日' ? 'align' : 'read')
  }, [concept.id])
```

Insert after line 62 (after the docContent loading useEffect).

- [ ] **Step 4: Typecheck**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add src/components/ConceptDetailPanel.tsx
git commit -m "feat: auto-switch to align tab for concepts due for review"
```

---

### Task 5: Trigger SM-2 review after alignment

**Files:**
- Modify: `src/components/AlignmentPanel.tsx`

- [ ] **Step 1: Add SM-2 call after alignment**

In the `handleAlign` callback, after `updateMastery` call (line 94), add:

```typescript
    // Trigger SM-2 review record based on alignment quality
    const coverage = stats.nodeCoverage || 0
    let quality: number
    if (coverage > 80) quality = 4
    else if (coverage > 50) quality = 3
    else quality = 2
    useKnowledgeGraphStore.getState().startReview(concept.id, quality)
```

The full `handleAlign` becomes:

```typescript
  const handleAlign = useCallback(() => {
    if (!userText.trim() || !originalContent) return
    const r = compareTexts(userText, originalContent, allConcepts, concept.id)
    setResult(r)
    setHasAligned(true)
    const { stats } = r
    const score = Math.round(
      (stats.nodeCoverage || 0) * 0.6 + (stats.nodePrecision || 0) * 0.4
    )
    useKnowledgeGraphStore.getState().updateMastery(concept.id, score)
    // Trigger SM-2 review
    const coverage = stats.nodeCoverage || 0
    let quality: number
    if (coverage > 80) quality = 4
    else if (coverage > 50) quality = 3
    else quality = 2
    useKnowledgeGraphStore.getState().startReview(concept.id, quality)
  }, [userText, originalContent, allConcepts, concept.id])
```

- [ ] **Step 2: Verify the import is sufficient**

`startReview` is already used from `useKnowledgeGraphStore` which is already imported. No new imports needed.

- [ ] **Step 3: Typecheck**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/components/AlignmentPanel.tsx
git commit -m "feat: trigger SM-2 review after alignment completion"
```

---

### Task 6: Add review due queue to SummaryPanel

**Files:**
- Modify: `src/components/SummaryPanel.tsx`

- [ ] **Step 1: Import and compute due concepts**

Add imports:
```typescript
import { getDueConcepts, type ReviewBadgeType } from '../utils/knowledgeGraph'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
```

Add review data computation:
```typescript
  const reviewRecords = useKnowledgeGraphStore(s => s.reviewRecords)
  const dueConcepts = useMemo(
    () => getDueConcepts(concepts, reviewRecords).filter(d => d.badge.urgency <= 2),
    [concepts, reviewRecords]
  )

  const onTrackCount = useMemo(() => {
    let total = 0
    let onTime = 0
    reviewRecords.forEach(r => {
      if (r.review_count > 0) {
        total++
        const diff = Math.ceil((new Date(r.next_review).getTime() - Date.now()) / (1000 * 60 * 60 * 24))
        if (diff >= Math.ceil(r.interval * 0.5)) onTime++ // reviewed roughly on schedule
      }
    })
    return { total, onTime }
  }, [reviewRecords])
```

- [ ] **Step 2: Add review queue section**

After the longest paths SectionCard (before closing div), add:

```typescript
        <SectionCard
          title="📅 复习待办"
          count={dueConcepts.length}
          expanded={expandedSection === 'review'}
          onToggle={() => toggleSection('review')}
        >
          <div className="space-y-0.5">
            {dueConcepts.map((d, i) => (
              <button
                key={d.concept.id}
                className="flex items-center gap-2 px-2 py-1.5 text-xs hover:bg-gray-50 rounded cursor-pointer w-full text-left"
                onClick={() => onNavigate?.(d.concept.id)}
              >
                <span className="text-base shrink-0">{d.badge.text}</span>
                <div className="flex-1 min-w-0">
                  <span className="font-medium text-gray-700 truncate block">{d.concept.title}</span>
                </div>
                <span className={`shrink-0 text-[10px] ${
                  d.badge.urgency === 0 ? 'text-red-500 font-medium' : 'text-gray-400'
                }`}>
                  {d.daysUntilReview <= 0 ? `过期 ${Math.abs(d.daysUntilReview)} 天` : `${d.daysUntilReview} 天后`}
                </span>
              </button>
            ))}
            {dueConcepts.length === 0 && (
              <p className="text-xs text-gray-400 text-center py-2">暂无待复习概念</p>
            )}
          </div>
          {onTrackCount.total > 0 && (
            <div className="mt-2 pt-2 border-t border-gray-100">
              <div className="flex items-center justify-between text-[10px] text-gray-500 mb-1">
                <span>在轨率</span>
                <span>{Math.round((onTrackCount.onTime / onTrackCount.total) * 100)}% ({onTrackCount.onTime}/{onTrackCount.total})</span>
              </div>
              <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    onTrackCount.onTime / onTrackCount.total >= 0.8
                      ? 'bg-emerald-500'
                      : onTrackCount.onTime / onTrackCount.total >= 0.5
                      ? 'bg-amber-500'
                      : 'bg-red-400'
                  }`}
                  style={{ width: `${(onTrackCount.onTime / onTrackCount.total) * 100}%` }}
                />
              </div>
            </div>
          )}
        </SectionCard>
```

- [ ] **Step 3: Typecheck**

Run: `npx tsc --noEmit`
Expected: No errors. If `ReviewBadgeType` is not exported, update the import in `knowledgeGraph.ts`:

```typescript
export type { ReviewBadgeType } from '../utils/knowledgeGraph'
```

Actually just import it directly from the file where it's defined.

- [ ] **Step 4: Commit**

```bash
git add src/components/SummaryPanel.tsx
git commit -m "feat: add review due queue section to summary panel"
```
