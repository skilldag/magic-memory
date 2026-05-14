# 聚焦视图深度控制 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to select the depth (1/2/3/All) of related concepts shown in focus view, replacing the current hardcoded depth=1 behavior.

**Architecture:** BFS traversal from the selected concept using Cytoscape.js `connectedEdges().connectedNodes()` to collect nodes at configurable depth. A button group in the top toolbar controls the depth. The existing focus effect's node display logic is reused — only the `relatedNodeIds` set calculation changes.

**Tech Stack:** React 19, TypeScript, Cytoscape.js, Zustand, Tailwind CSS

---

### Task 1: Add BFS function and focusDepth prop to KnowledgeGraph

**Files:**
- Modify: `src/components/KnowledgeGraph.tsx` (lines 30-31 insert function, line 67 add prop, lines 70-97 destructure, lines 690-693 + 723-724 + 752 replace depth logic)

- [ ] **Step 1: Add `getNodesAtDepth` BFS function after `calcAdaptiveLayoutParams`**

Insert between line 31 (closing `}` of `calcAdaptiveLayoutParams`) and line 33 (blank + interface):

```typescript
function getNodesAtDepth(
  cy: Core,
  centerNodeId: string,
  depth: number
): string[] {
  const visited = new Set<string>([centerNodeId])
  let current = [centerNodeId]

  for (let level = 0; level < depth; level++) {
    const next: string[] = []
    for (const id of current) {
      const node = cy.getElementById(id)
      if (!node.length) continue
      node.connectedEdges().connectedNodes().forEach(n => {
        if (!visited.has(n.id())) {
          visited.add(n.id())
          next.push(n.id())
        }
      })
    }
    current = next
  }

  return [...visited]
}
```

- [ ] **Step 2: Add `focusDepth` prop to `KnowledgeGraphProps` interface**

Add after line 66 (`reviewMode?: boolean`):

```typescript
  // Focus depth: number of levels of related nodes to show (default 1, Infinity = all)
  focusDepth?: number
```

- [ ] **Step 3: Destructure `focusDepth` in component params**

Add after line 95 (`reviewMode = false,`):

```typescript
  focusDepth = 1,
```

- [ ] **Step 4: Replace focus effect's depth logic (lines 690-693)**

Replace lines 690-693 (the old hardcoded neighbor collection):

```typescript
    const selectedNode = cy.getElementById(selectedConcept.id)

    // Collect related nodes at configurable depth via BFS
    let relatedNodeIds: Set<string>
    if (focusDepth === Infinity) {
      relatedNodeIds = new Set(cy.nodes().map(n => n.id()))
    } else {
      const focusIds = getNodesAtDepth(cy, selectedConcept.id, focusDepth)
      relatedNodeIds = new Set(focusIds)
    }
    const neighborNodes = cy.collection(
      [...relatedNodeIds].filter(id => id !== selectedConcept.id).map(id => cy.getElementById(id))
    )
```

Note: `neighborNodes` is still needed by the layout section below (lines 721, 723, 742). We reconstruct it from the `relatedNodeIds` set minus the center node.

- [ ] **Step 5: Add `focusDepth` to effect dependency array (line 752)**

Change line 752 from:

```typescript
  }, [selectedConcept, focusEnabled, focusedNodeIds, structuralKey, conceptMastery])
```

to:

```typescript
  }, [selectedConcept, focusEnabled, focusedNodeIds, structuralKey, conceptMastery, focusDepth])
```

- [ ] **Step 6: Verify with LSP diagnostics**

Run: LSP diagnostics on `src/components/KnowledgeGraph.tsx`

Expected: Clean (no errors in the file). Pre-existing errors in worktree files are unrelated.

---

### Task 2: Add depth selector UI to KnowledgeGraphView

**Files:**
- Modify: `src/components/KnowledgeGraphView.tsx` (add state, add UI, pass prop)

- [ ] **Step 1: Add `focusDepth` state**

Find the state declarations section (around line 65-70, after `const [bannerDismissed, setBannerDismissed] = useState(false)`). Add:

```typescript
  const [focusDepth, setFocusDepth] = useState<number>(1)
```

- [ ] **Step 2: Add depth selector buttons in the toolbar**

Find the toolbar section (lines 306-324). After the `<span>` that shows "聚焦: xxx" (line 321 closing `</span>`), add the depth selector:

```tsx
            {selectedConcept && (
              <div className="flex items-center gap-1 ml-1 shrink-0">
                <span className="text-[11px] text-gray-400 mr-0.5">深度</span>
                {[
                  { value: 1, label: '1' },
                  { value: 2, label: '2' },
                  { value: 3, label: '3' },
                  { value: Infinity, label: '∞' },
                ].map(d => (
                  <button
                    key={d.label}
                    onClick={() => setFocusDepth(d.value)}
                    className={`w-6 h-6 rounded text-xs font-medium transition-colors ${
                      focusDepth === d.value
                        ? 'bg-blue-500 text-white'
                        : 'bg-white text-gray-600 hover:bg-gray-100 border border-gray-200'
                    }`}
                  >
                    {d.label}
                  </button>
                ))}
              </div>
            )}
```

- [ ] **Step 3: Pass `focusDepth` prop to KnowledgeGraph**

Find the `<KnowledgeGraph` component usage (around line 439). Add the prop after `focusEnabled={true}`:

```tsx
            focusDepth={focusDepth}
```

- [ ] **Step 4: Verify with LSP diagnostics**

Run: LSP diagnostics on `src/components/KnowledgeGraphView.tsx`

Expected: Clean (no errors in this file).

---

### Task 3: Verify the feature works end-to-end

- [ ] **Step 1: Start the dev server**

Run: `npm run dev` in the project root (or `make dev` — check Makefile for the correct command).

Expected: Dev server starts without errors.

- [ ] **Step 2: Manual verification**

1. Open the app in browser
2. Click a concept node to enter focus view → default shows only direct neighbors (depth 1)
3. Verify the depth selector appears in the toolbar: [1] [2] [3] [∞], with [1] highlighted
4. Click [2] → graph should now show concepts at 2 hops distance + auto re-layout
5. Click [3] → show 3 hops
6. Click [∞] → show all nodes (same as unfocused but with selected node highlighted)
7. Click [1] again → back to depth 1
8. Deselect concept → depth selector disappears
9. Re-select a different concept → depth defaults to 1

- [ ] **Step 3: Edge case verification**

1. Select an isolated concept (no edges) → all depths show only that one node
2. Select a concept with only 1 neighbor → depth 2 should not reveal more nodes
3. Click depth buttons rapidly → each click triggers a re-layout, last one wins

---

### Verification Summary

| Check | Command |
|-------|---------|
| LSP diagnostics (KG.tsx) | `lsp_diagnostics(filePath="src/components/KnowledgeGraph.tsx")` |
| LSP diagnostics (KGView.tsx) | `lsp_diagnostics(filePath="src/components/KnowledgeGraphView.tsx")` |
| Build | `npm run build` or `npx tsc --noEmit` |
