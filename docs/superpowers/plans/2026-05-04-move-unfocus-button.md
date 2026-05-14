# Move Unfocus Button to Graph Area — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the × "取消选择" unfocus button from the right-side concept detail panel to a floating top bar on the left-side knowledge graph area.

**Architecture:** Two files are modified: `ConceptDetailPanel.tsx` (remove the × button) and `KnowledgeGraphView.tsx` (add a floating bar that shows when a concept is focused). The same `onDeselect` callback is reused — no state changes needed.

**Tech Stack:** React 19 + TypeScript + Tailwind CSS 4

---

### Task 1: Remove × button from ConceptDetailPanel

**Files:**
- Modify: `src/components/ConceptDetailPanel.tsx:280-285`

- [ ] **Step 1: Remove the × button JSX**

Remove lines 280-285 in `ConceptDetailPanel.tsx`:

```tsx
// BEFORE (lines 280-285):
          <div className="flex items-center gap-1">
            <button onClick={onDeselect} className="shrink-0 p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600" title="取消选择">
              <svg width={16} height={16} className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

// AFTER:
          <div className="flex items-center gap-1">
            {/* unfocus button moved to graph area */}
          </div>
```

Keep the wrapping `<div className="flex items-center gap-1">` empty for now (or remove it entirely — it's harmless to leave). The simplest change: remove only the `<button>` block and leave the empty div.

- [ ] **Step 2: Verify no broken references**

The `onDeselect` prop is still used — only the JSX button is removed. Verify by running:

```bash
npx tsc --noEmit --pretty
```

Expected: No TypeScript errors (the `onDeselect` prop type in `ConceptDetailPanelProps` is still referenced, but that's fine — the prop is still used by `KnowledgeGraphView` for other purposes).

- [ ] **Step 3: Commit**

```bash
git add src/components/ConceptDetailPanel.tsx
git commit -m "refactor: remove × unfocus button from concept detail panel"
```

### Task 2: Add floating top bar in KnowledgeGraphView

**Files:**
- Modify: `src/components/KnowledgeGraphView.tsx` (add floating bar inside `graphContainerRef` div)

- [ ] **Step 1: Add floating bar JSX**

In `KnowledgeGraphView.tsx`, inside the `graphContainerRef` div (which has `className="min-w-0 relative flex flex-col"`), add a floating bar before the main content. Insert it right after the loading progress bar (after line 289).

Add this JSX block:

```tsx
{/* 聚焦模式顶部悬浮条 */}
{selectedConcept && !processMode && (
  <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-4 py-2 bg-white/80 backdrop-blur-sm border-b border-blue-200 shadow-sm">
    <div className="flex items-center gap-2">
      <span className="w-2 h-2 rounded-full bg-green-500" />
      <span className="text-sm font-medium text-gray-700">
        聚焦: <span className="text-blue-600">{selectedConcept.title}</span>
      </span>
    </div>
    <button
      onClick={() => {
        setSelectedConceptId(null);
        useKnowledgeGraphStore.setState({ selectedConcept: null });
      }}
      className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 hover:text-gray-800 transition-colors"
    >
      <svg width={14} height={14} className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
      退出聚焦
    </button>
  </div>
)}
```

Placement: Insert after the loading progress bar block (which ends at line 290), before the `showProjectList` condition (line 291).

- [ ] **Step 2: Verify with type check**

```bash
npx tsc --noEmit --pretty
```

Expected: No TypeScript errors. The `useKnowledgeGraphStore` is already imported at the top of the file (line 11). `selectedConcept` and `processMode` are already in scope.

- [ ] **Step 3: Commit**

```bash
git add src/components/KnowledgeGraphView.tsx
git commit -m "feat: add floating unfocus bar to knowledge graph area"
```

### Task 3: Verify end-to-end

- [ ] **Step 1: Run build**

```bash
npm run build
```

Expected: Build succeeds with no errors.

- [ ] **Step 2: Manual verification checklist**

1. Open the app in dev mode (`npm run dev`)
2. Navigate to the knowledge graph view
3. **With no concept selected**: Verify no floating bar appears at top of graph
4. **Click a concept node**: Verify a floating bar appears at the top of the graph showing "聚焦: {concept name}" and an "退出聚焦" button
5. **Click "退出聚焦"**: Verify the graph exits focus mode and the bar disappears
6. **Click a concept and double-click to enter ProcessCanvas**: Verify no floating bar appears in ProcessCanvas mode
7. **Verify right panel no longer shows × button** in the concept detail header
8. **Verify right panel deselect still works via "退出聚焦" button** (the onDeselect path remains intact)
