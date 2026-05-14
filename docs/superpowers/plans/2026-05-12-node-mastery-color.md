# Node Mastery Color Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Level-based node colors with heatmap colors reflecting semantic alignment scores.

**Architecture:** Each semantic alignment produces a combined score (nodeCoverage×60% + nodePrecision×40%), stored in a `conceptMastery` Map in Zustand. The KnowledgeGraph reads mastery scores and applies colors via `getMasteryColor()`. Data is persisted through the existing Graph Server API.

**Tech Stack:** React 19 + TypeScript + Zustand + Cytoscape

---

### Task 1: Add MasteryRecord type

**Files:**
- Modify: `src/types/index.ts` (after line 177, before SuggestionItem)

- [ ] **Step 1: Add MasteryRecord interface**

Add after `UserAnnotation` interface (line 187):

```typescript
export interface MasteryRecord {
  conceptId: string
  score: number          // 0-100 composite mastery score
  lastAligned: string    // ISO date string of last alignment
  alignmentCount: number // how many times alignment was run
}
```

- [ ] **Step 2: Verify no type errors**

Run: `npx tsc --noEmit --pretty 2>&1 | head -30`
Expected: No errors related to MasteryRecord

- [ ] **Step 3: Commit**

```bash
git add src/types/index.ts
git commit -m "feat: add MasteryRecord type for node mastery tracking"
```

---

### Task 2: Add MASTERY_COLORS to graph constants

**Files:**
- Modify: `src/constants/graph.ts`

- [ ] **Step 1: Replace LEVEL_COLORS with MASTERY_COLORS**

Replace the entire file content:

```typescript
export const EDGE_COLORS: Record<string, string> = {
  depends_on: '#ef4444',
  leads_to: '#10b981',
  related: '#6b7280',
}

export const MASTERY_COLORS: Record<string, string> = {
  unaligned: '#d1d5db',  // gray - never aligned
  weak:      '#ef4444',   // red - score < 40
  partial:   '#f59e0b',   // amber - 40 ≤ score < 70
  good:      '#10b981',   // green - 70 ≤ score < 90
  mastered:  '#059669',   // deep green - score ≥ 90
}

export function getMasteryColor(score: number | undefined): string {
  if (score === undefined) return MASTERY_COLORS.unaligned
  if (score >= 90) return MASTERY_COLORS.mastered
  if (score >= 70) return MASTERY_COLORS.good
  if (score >= 40) return MASTERY_COLORS.partial
  return MASTERY_COLORS.weak
}
```

- [ ] **Step 2: Verify**

Check that `getMasteryColor` is exported and the file has no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add src/constants/graph.ts
git commit -m "feat: replace LEVEL_COLORS with MASTERY_COLORS and getMasteryColor()"
```

---

### Task 3: Add conceptMastery state to store

**Files:**
- Modify: `src/store/knowledgeGraphStore.ts`

- [ ] **Step 1: Add conceptMastery to interface and initial state**

In the `KnowledgeGraphStore` interface, add before `updateConceptContent` (after line 58):

```typescript
conceptMastery: Map<string, MasteryRecord>
updateMastery: (conceptId: string, score: number) => void
```

In the initial state (after line 77), add:

```typescript
conceptMastery: new Map(),
```

- [ ] **Step 2: Add import for MasteryRecord**

At line 2, add `MasteryRecord` to the import from `'../types'`:

```typescript
import type { Concept, ConceptEdge, ReviewRecord, UserAnnotation, ProcessChain, ProcessState, MasteryRecord } from '../types'
```

- [ ] **Step 3: Implement updateMastery action**

Add after `updateConceptContent` (after line 189):

```typescript
updateMastery: (conceptId, score) => {
  const { conceptMastery } = get()
  const existing = conceptMastery.get(conceptId)
  const record: MasteryRecord = {
    conceptId,
    score,
    lastAligned: new Date().toISOString(),
    alignmentCount: (existing?.alignmentCount ?? 0) + 1,
  }
  const newMap = new Map(conceptMastery)
  newMap.set(conceptId, record)
  set({ conceptMastery: newMap })
},
```

- [ ] **Step 4: Load mastery data from Graph Server in loadProjectGraph**

In `loadProjectGraph`, after the `data` is fetched (after line 100), parse mastery data:

```typescript
set(state => {
  // ...existing code for selectedConcept preservation...
  return {
    concepts: newConcepts,
    edges: data.edges || [],
    conceptMastery: data.mastery
      ? new Map(Object.entries(data.mastery).map(([k, v]) => [k, v as MasteryRecord]))
      : state.conceptMastery,
    selectedConcept: preserved,
    isLoading: false,
    loadingProgress: 100,
  }
})
```

- [ ] **Step 5: Persist mastery data to server**

In `persistToServer`, replace the body sent:

```typescript
const { activeProjectId, concepts, edges, conceptMastery } = get()
// ...rest of function...
body: JSON.stringify({
  concepts,
  edges,
  mastery: Object.fromEntries(conceptMastery),
}),
```

- [ ] **Step 6: Commit**

```bash
git add src/store/knowledgeGraphStore.ts
git commit -m "feat: add conceptMastery state with updateMastery action and persistence"
```

---

### Task 4: Update KnowledgeGraph to color nodes by mastery

**Files:**
- Modify: `src/components/KnowledgeGraph.tsx`

- [ ] **Step 1: Add mastery prop to interface**

Add after `focusedNodeIds` (around line 49):

```typescript
// Mastery data for node coloring: conceptId → score
conceptMastery?: Map<string, { score: number }>
```

- [ ] **Step 2: Add master to default value destructuring**

Add `conceptMastery` to the props destructuring (around line 73):

```typescript
conceptMastery,
```

- [ ] **Step 3: Update import — replace LEVEL_COLORS with MASTERY_COLORS + getMasteryColor**

Change line 6 from:
```typescript
import { LEVEL_COLORS, EDGE_COLORS } from '../constants/graph'
```
to:
```typescript
import { MASTERY_COLORS, EDGE_COLORS, getMasteryColor } from '../constants/graph'
```

- [ ] **Step 4: Update initial node style — replace Level-based with mastery-based**

In the initial style (lines 234-244), replace the Level-based selectors:

```typescript
// REMOVE these three selectors (lines 234-244):
{
  selector: 'node[level="1"]',
  style: { 'background-color': LEVEL_COLORS[1] }
},
{
  selector: 'node[level="2"]',
  style: { 'background-color': LEVEL_COLORS[2] }
},
{
  selector: 'node[level="3"]',
  style: { 'background-color': LEVEL_COLORS[3] }
},
```

- [ ] **Step 5: Add node creation time — attach mastery data**

In the initial elements creation (line 193), after `category: c.category`, add:

```typescript
mastery: conceptMastery?.get(c.id)?.score,
```

This will pass mastery score as node data on first render.

- [ ] **Step 6: Update focus mode style references (3 locations)**

In the focus effect (around lines 563, 598, 635), replace each occurrence of:
```typescript
'background-color': isFirst ? '#f59e0b' : (LEVEL_COLORS[Number(n.data('level'))] || '#3b82f6'),
```
with:
```typescript
'background-color': isFirst ? '#f59e0b' : (getMasteryColor(n.data('mastery'))),
```

There are 3 occurrences to replace:
1. Line ~563 (precision focus)
2. Line ~598 (no focus/unfocus)
3. Line ~635 (focus mode)

- [ ] **Step 7: Add mastery ref for incremental updates**

Add `conceptMasteryRef` next to the other refs (after line 104):

```typescript
const conceptMasteryRef = useRef(conceptMastery)
conceptMasteryRef.current = conceptMastery
```

- [ ] **Step 8: Update incremental node addition — attach mastery data**

In the incremental update effect (line 492), change the node data to include mastery:

```typescript
data: {
  id: c.id,
  label: c.title,
  level: c.level,
  category: c.category,
  mastery: conceptMasteryRef.current?.get(c.id)?.score,
}
```

- [ ] **Step 9: Update legend — replace Level colors with mastery legend**

Replace the level legend section (lines 781-794) with:

```tsx
<div className="border-t border-gray-200 my-1.5" />
<div className="font-medium text-gray-700 mb-1">掌握程度</div>
<div className="flex items-center gap-2">
  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: MASTERY_COLORS.unaligned }} />
  <span className="text-gray-600">未对齐</span>
</div>
<div className="flex items-center gap-2">
  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: MASTERY_COLORS.weak }} />
  <span className="text-gray-600">薄弱 (&lt;40%)</span>
</div>
<div className="flex items-center gap-2">
  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: MASTERY_COLORS.partial }} />
  <span className="text-gray-600">部分掌握 (40-70%)</span>
</div>
<div className="flex items-center gap-2">
  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: MASTERY_COLORS.good }} />
  <span className="text-gray-600">良好 (70-90%)</span>
</div>
<div className="flex items-center gap-2">
  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: MASTERY_COLORS.mastered }} />
  <span className="text-gray-600">精通 (&gt;90%)</span>
</div>
```

- [ ] **Step 10: Commit**

```bash
git add src/components/KnowledgeGraph.tsx
git commit -m "feat: color knowledge graph nodes by mastery score instead of level"
```

---

### Task 5: Pass conceptMastery from KnowledgeGraphView

**Files:**
- Modify: `src/components/KnowledgeGraphView.tsx`

- [ ] **Step 1: Get conceptMastery from store**

Add after the existing store selectors (around line 26):

```typescript
const conceptMastery = useKnowledgeGraphStore(s => s.conceptMastery)
```

- [ ] **Step 2: Pass conceptMastery to KnowledgeGraph component**

In the `<KnowledgeGraph>` JSX (around line 401), add the prop:

```tsx
<KnowledgeGraph
  concepts={concepts} edges={edges} selectedConcept={selectedConcept}
  conceptMastery={conceptMastery}
  focusEnabled={true}
  ...
```

- [ ] **Step 3: Commit**

```bash
git add src/components/KnowledgeGraphView.tsx
git commit -m "feat: pass conceptMastery from view to KnowledgeGraph component"
```

---

### Task 6: Call updateMastery from AlignmentPanel

**Files:**
- Modify: `src/components/AlignmentPanel.tsx`

- [ ] **Step 1: Add import for useKnowledgeGraphStore**

At the top, the existing import already includes `useKnowledgeGraphStore`. Verify it's there:

```typescript
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
```
✅ Already imported line 10.

- [ ] **Step 2: Call updateMastery after alignment**

In `handleAlign` (line 76-81), after `setHasAligned(true)`, add:

```typescript
const { stats } = r
const score = Math.round(
  (stats.nodeCoverage || 0) * 0.6 + (stats.nodePrecision || 0) * 0.4
)
useKnowledgeGraphStore.getState().updateMastery(concept.id, score)
```

So the complete handler becomes:

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
}, [userText, originalContent, allConcepts, concept.id])
```

- [ ] **Step 3: Show mastery score feedback after alignment**

After the alignment result area (after line 136, inside the `hasAligned && result` block), add a score feedback line:

```tsx
{hasAligned && result && (
  <div className="space-y-4 pt-2 border-t border-gray-100">
    {/* ...existing stats code... */}
    
    {/* Add score feedback */}
    <div className="p-2.5 rounded-lg border border-blue-200 bg-blue-50/50 flex items-center justify-between">
      <span className="text-xs text-blue-800 font-medium">本次掌握分</span>
      <span className="text-sm font-bold text-blue-700">
        {Math.round((result.stats.nodeCoverage || 0) * 0.6 + (result.stats.nodePrecision || 0) * 0.4)}/100
      </span>
    </div>
    
    {/* ...rest of the result content... */}
```

Place this right before the fuzzyMatches section (before the `{result.fuzzyMatches.length > 0 && (` block).

- [ ] **Step 4: Commit**

```bash
git add src/components/AlignmentPanel.tsx
git commit -m "feat: update node mastery score after semantic alignment"
```

---

### Task 7: Verify build passes

**Files:** None

- [ ] **Step 1: Run TypeScript check**

Run: `npx tsc --noEmit --pretty 2>&1 | head -40`
Expected: No errors related to our changes (pre-existing Bun/module errors in `.worktrees/` and `magic-memory/` directories may appear but should be unrelated).

- [ ] **Step 2: Run vite build**

Run: `npx vite build 2>&1 | tail -20`
Expected: Build succeeds

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: resolve type errors after mastery color implementation"
```
