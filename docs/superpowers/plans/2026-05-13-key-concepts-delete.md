# KEY CONCEPTS 条目删除 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to temporarily ignore or permanently delete individual KEY CONCEPTS terms from the semantic alignment panel.

**Architecture:** Extend the existing AlignmentPanel with hover-action buttons on NodeRow components. Ignored terms are stored in `alignmentDrafts` in the zustand store (per-concept). Permanent deletion modifies the document content via `updateConceptContent()` and persists to server.

**Tech Stack:** React 19 + TypeScript + Zustand + Tailwind CSS 4

---

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `src/utils/alignment.ts` | KEY CONCEPTS text manipulation | Add `removeKeyConceptFromContent()` function |
| `src/components/AlignmentPanel.tsx` | Alignment UI with node list | Enhance NodeRow, add ignored section, add delete/ignore workflows |
| `src/store/knowledgeGraphStore.ts` | Global state | Extend `alignmentDrafts` value type to include `ignoredTerms` |

---

### Task 1: Add `removeKeyConceptFromContent()` to alignment.ts

**Files:**
- Modify: `src/utils/alignment.ts` (append at end, before last line)

- [ ] **Step 1: Write the function**

Add after the `alignTokenSets` function (before file end):

```typescript
/**
 * Remove a single term from the KEY CONCEPTS paragraph in the document content.
 * Returns the modified content, or null if the term was not found.
 */
export function removeKeyConceptFromContent(
  content: string,
  termToRemove: string
): string | null {
  const lines = content.split('\n')
  let inSection = false
  let modified = false

  const result = lines.map(line => {
    const trimmed = line.trim()
    if (/^#*\s*KEY CONCEPTS:?\s*$/i.test(trimmed)) {
      inSection = true
      return line
    }
    if (inSection) {
      if (trimmed.startsWith('#') || (trimmed === '' && modified)) {
        inSection = false
        return line
      }
      if (trimmed) {
        // Split by whitespace, filter out the exact term
        const terms = line.split(/\s+/)
        const filtered = terms.filter(t => t !== termToRemove)
        if (filtered.length !== terms.length) {
          modified = true
        }
        return filtered.join(' ')
      }
    }
    return line
  })

  return modified ? result.join('\n') : null
}
```

- [ ] **Step 2: Verify function logic**

Check with these test inputs mentally:
- content with `KEY CONCEPTS: term1 term2 term3` → remove `term2` → `KEY CONCEPTS: term1 term3`
- content where term doesn't exist → returns null
- content with no KEY CONCEPTS section → returns null
- multi-line KEY CONCEPTS section → handles correctly

- [ ] **Step 3: Commit**

```bash
git add src/utils/alignment.ts
git commit -m "feat: add removeKeyConceptFromContent utility function"
```

---

### Task 2: Extend alignmentDrafts type in knowledgeGraphStore.ts

**Files:**
- Modify: `src/store/knowledgeGraphStore.ts`

- [ ] **Step 1: Extend the alignmentDrafts value type**

Find the `alignmentDrafts` type definition in the store interface (around line 63-72):

```typescript
// Before:
alignmentDrafts: Map<string, {
  userText: string
  hasAligned: boolean
  result: GraphAlignmentResult | null
}>

// After:
alignmentDrafts: Map<string, {
  userText: string
  hasAligned: boolean
  result: GraphAlignmentResult | null
  ignoredTerms: string[]    // ← ADD
}>
```

- [ ] **Step 2: Add default for ignoredTerms in setAlignmentDraft if missing**

Find the `setAlignmentDraft` action. No change needed — the default value will be an empty array `[]` when first used in AlignmentPanel.

- [ ] **Step 3: Commit**

```bash
git add src/store/knowledgeGraphStore.ts
git commit -m "feat: extend alignmentDrafts type with ignoredTerms field"
```

---

### Task 3: Enhance NodeRow with hover delete/ignore buttons

**Files:**
- Modify: `src/components/AlignmentPanel.tsx`

- [ ] **Step 1: Add `onIgnore` and `onDelete` callbacks to NodeRow props**

```typescript
interface NodeRowProps {
  node: AlignedNodePair
  onIgnore?: (node: AlignedNodePair) => void
  onDelete?: (node: AlignedNodePair) => void
  showActions?: boolean  // true for 'missing' nodes only
}
```

- [ ] **Step 2: Update NodeRow component to show action buttons on hover**

Replace the existing NodeRow component:

```typescript
function NodeRow({ node, onIgnore, onDelete, showActions }: NodeRowProps) {
  const dot = { matched: 'bg-emerald-500', missing: 'bg-amber-500', extra: 'bg-gray-400' }
  const bg  = { matched: 'border-emerald-200 bg-emerald-50/50', missing: 'border-amber-200 bg-amber-50', extra: 'border-gray-200 bg-gray-50' }
  const lb  = { matched: '已理解', missing: '未提及', extra: '多余' }
  const [hovered, setHovered] = useState(false)

  return (
    <div
      className={`rounded-lg border p-2.5 text-xs ${bg[node.status]}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center gap-1.5">
        <span className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${dot[node.status]}`} />
        <span className="font-medium truncate">{node.label}</span>
        {node.isKnownConcept && <span className="text-[9px] text-blue-500 font-medium">KG</span>}
        <span className={`ml-auto px-1 py-0.5 rounded text-[9px] font-medium ${
          node.status === 'matched' ? 'bg-emerald-100 text-emerald-700' :
          node.status === 'missing' ? 'bg-amber-100 text-amber-700' :
          'bg-gray-100 text-gray-500'
        }`}>{lb[node.status]}</span>
        {/* Action buttons: show on hover for missing nodes */}
        {hovered && showActions && (
          <div className="flex items-center gap-0.5 ml-1">
            <button
              onClick={(e) => { e.stopPropagation(); onIgnore?.(node) }}
              className="p-0.5 rounded hover:bg-gray-200 text-gray-400 hover:text-gray-600 transition-colors"
              title="临时忽略此条目"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete?.(node) }}
              className="p-0.5 rounded hover:bg-red-100 text-gray-400 hover:text-red-500 transition-colors"
              title="从原文永久删除"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Update the render loop to pass new props**

Find where `NodeRow` is rendered (around line 207-208) and update:

```typescript
{showTab === 'nodes' && (
  <div className="space-y-1.5">
    {result.nodes
      .filter(n => !ignoredTerms.includes(n.label))  // ← filter out ignored
      .sort((a, b) => ({ matched: 0, missing: 1, extra: 2 }[a.status] - { matched: 0, missing: 1, extra: 2 }[b.status]))
      .map(n => (
        <NodeRow
          key={n.nodeId}
          node={n}
          showActions={n.status === 'missing'}
          onIgnore={handleIgnoreNode}
          onDelete={handleDeleteNode}
        />
      ))
    }
  </div>
)}
```

- [ ] **Step 4: Commit**

```bash
git add src/components/AlignmentPanel.tsx
git commit -m "feat: add hover ignore/delete buttons to NodeRow in alignment panel"
```

---

### Task 4: Add ignored terms state management and handlers

**Files:**
- Modify: `src/components/AlignmentPanel.tsx`

- [ ] **Step 1: Load ignoredTerms from store draft**

Inside `AlignmentPanel`, after existing state declarations (around line 58-63), add ignoredTerms state:

```typescript
const [ignoredTerms, setIgnoredTerms] = useState<string[]>(draft?.ignoredTerms ?? [])

// Sync ignoredTerms to draft whenever it changes
useEffect(() => {
  setAlignmentDraft(concept.id, {
    userText,
    hasAligned,
    result,
    ignoredTerms,
  })
}, [userText, hasAligned, result, ignoredTerms, concept.id, setAlignmentDraft])
```

Also update the existing `useEffect` sync at line 67 to include `ignoredTerms`:

```typescript
useEffect(() => {
  setAlignmentDraft(concept.id, { userText, hasAligned, result, ignoredTerms })
}, [userText, hasAligned, result, ignoredTerms, concept.id, setAlignmentDraft])
```

- [ ] **Step 2: Add handleIgnoreNode handler**

```typescript
const handleIgnoreNode = useCallback((node: AlignedNodePair) => {
  setIgnoredTerms(prev => {
    if (prev.includes(node.label)) return prev
    const next = [...prev, node.label]
    // Recompute result with updated ignored terms
    updateStats(next)
    return next
  })
}, [])
```

- [ ] **Step 3: Add handleDeleteNode handler**

```typescript
const [deleteConfirm, setDeleteConfirm] = useState<AlignedNodePair | null>(null)

const handleDeleteNode = useCallback((node: AlignedNodePair) => {
  setDeleteConfirm(node)
}, [])

const confirmDelete = useCallback(async () => {
  if (!deleteConfirm || !concept.content) return
  const newContent = removeKeyConceptFromContent(concept.content, deleteConfirm.label)
  if (!newContent) return
  // Update store and re-run alignment
  updateConceptContent(concept.id, newContent)
  setOriginalContent(newContent)
  setDeleteConfirm(null)
  // Re-run alignment with new content
  if (userText.trim()) {
    const r = compareTexts(userText, newContent, allConcepts, concept.id)
    setResult(r)
    // Update mastery and review
    const score = Math.round((r.stats.nodeCoverage || 0) * 0.6 + (r.stats.nodePrecision || 0) * 0.4)
    useKnowledgeGraphStore.getState().updateMastery(concept.id, score)
  }
}, [deleteConfirm, concept.content, concept.id, userText, allConcepts])
```

- [ ] **Step 4: Add delete confirmation dialog**

Add JSX somewhere in the return (after the main content, before the closing `</div>`):

```tsx
{/* Delete confirmation dialog */}
{deleteConfirm && (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setDeleteConfirm(null)}>
    <div className="bg-white rounded-xl shadow-xl p-5 max-w-sm mx-4" onClick={e => e.stopPropagation()}>
      <h4 className="text-sm font-semibold text-gray-900 mb-2">确认永久删除</h4>
      <p className="text-xs text-gray-600 mb-4">
        确定从原文 KEY CONCEPTS 段落中永久删除「<span className="font-medium text-gray-900">{deleteConfirm.label}</span>」吗？此操作会修改文档内容。
      </p>
      <div className="flex justify-end gap-2">
        <button onClick={() => setDeleteConfirm(null)} className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200">
          取消
        </button>
        <button onClick={confirmDelete} className="px-3 py-1.5 text-xs font-medium text-white bg-red-500 rounded-lg hover:bg-red-600">
          确认删除
        </button>
      </div>
    </div>
  </div>
)}
```

- [ ] **Step 5: Add updateStats helper**

```typescript
const updateStats = useCallback((ignored: string[]) => {
  if (!result) return
  // Filter out ignored terms and recalculate
  const visible = result.nodes.filter(n => !ignored.includes(n.label))
  const matchedCount = visible.filter(n => n.status === 'matched').length
  const missingCount = visible.filter(n => n.status === 'missing').length
  const extraCount = visible.filter(n => n.status === 'extra').length
  const total = result.originalNodeCount - (result.originalNodeCount - (matchedCount + missingCount))
  
  const newStats = {
    ...result.stats,
    matchedNodeCount: matchedCount,
    missingNodeCount: missingCount,
    extraNodeCount: extraCount,
    nodeCoverage: result.originalNodeCount > 0
      ? Math.round((matchedCount / result.originalNodeCount) * 100) : 0,
    nodePrecision: visible.length > 0
      ? Math.round((matchedCount / visible.length) * 100) : 0,
  }
  
  setResult({ ...result, stats: newStats })
}, [result])
```

Actually, let me rethink the stats recalculation. The `originalNodeCount` should remain unchanged to reflect the total in the original content. Only the visible counts should change. Let me adjust.

- [ ] **Step 6: Commit**

```bash
git add src/components/AlignmentPanel.tsx
git commit -m "feat: add ignore/delete handlers and confirmation dialog for KEY CONCEPTS"
```

---

### Task 5: Add ignored terms collapsible section

**Files:**
- Modify: `src/components/AlignmentPanel.tsx`

- [ ] **Step 1: Add ignored state and section to the JSX**

After the "原文有但你的描述中未出现的术语" (missing) section (around line 177-193), add:

```tsx
{/* Ignored terms section */}
{ignoredTerms.length > 0 && (
  <div className="p-3 rounded-lg border border-gray-200 bg-gray-50">
    <button
      onClick={() => setShowIgnored(!showIgnored)}
      className="flex items-center gap-1.5 w-full text-left"
    >
      <svg
        width={10} height={10}
        className={`text-gray-400 transition-transform ${showIgnored ? 'rotate-90' : ''}`}
        viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
      </svg>
      <h4 className="text-xs font-semibold text-gray-500">已忽略 ({ignoredTerms.length})</h4>
    </button>
    {showIgnored && (
      <div className="mt-2 space-y-1">
        {result?.nodes
          .filter(n => ignoredTerms.includes(n.label))
          .map(n => (
            <div key={n.nodeId} className="flex items-center justify-between px-2 py-1 rounded bg-white border border-gray-100">
              <div className="flex items-center gap-1.5">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-300" />
                <span className="text-xs text-gray-500">{n.label}</span>
              </div>
              <button
                onClick={() => {
                  setIgnoredTerms(prev => prev.filter(t => t !== n.label))
                }}
                className="text-[10px] text-blue-500 hover:text-blue-700 font-medium"
              >
                恢复
              </button>
            </div>
          ))
        }
      </div>
    )}
  </div>
)}
```

Also add state for showIgnored:
```typescript
const [showIgnored, setShowIgnored] = useState(true)
```

- [ ] **Step 2: Commit**

```bash
git add src/components/AlignmentPanel.tsx
git commit -m "feat: add collapsible ignored terms section with restore"
```

---

### Task 6: LSP diagnostics and build verification

**Files:**
- Verify: `src/components/AlignmentPanel.tsx`
- Verify: `src/utils/alignment.ts`
- Verify: `src/store/knowledgeGraphStore.ts`

- [ ] **Step 1: Run LSP diagnostics on changed files**

```bash
# Check for any type errors
```

- [ ] **Step 2: Build and verify**

```bash
npm run build
```

Expected: Build passes with no errors.

- [ ] **Step 3: Final commit if fixes were needed**

```bash
git add -A
git commit -m "fix: lint and type fixes after KEY CONCEPTS delete feature"
```
