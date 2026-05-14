# Focus Auto-Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When entering focus view (selecting a concept), automatically run adaptive fcose layout on the visible (neighbor) nodes so they rearrange optimally for the container.

**Architecture:** In the existing focus effect inside `KnowledgeGraph.tsx`, append ~25 lines after the existing `cy.fit()` call. Use the existing `calcAdaptiveLayoutParams` function to compute parameters based on visible node count, run fcose layout with animation, then fit to visible nodes on completion.

**Tech Stack:** React 19, TypeScript, Cytoscape.js + fcose

---

### Task 1: Add auto-layout to focus effect

**Files:**
- Modify: `src/components/KnowledgeGraph.tsx` ~line 640

- [ ] **Step 1: Locate the insertion point**

Open `src/components/KnowledgeGraph.tsx` and find the focus effect — it starts around line 532 with `useEffect(() => { ... }, [selectedConcept, focusEnabled, focusedNodeIds, structuralKey])`.

Inside this effect, find the branch that handles `selectedConcept && focusEnabled` (line 609). At the end of this branch, immediately after `cy.fit(neighborNodes.union(selectedNode), 60)` (line 640), add the auto-layout code.

- [ ] **Step 2: Add the auto-layout block**

After line 640 (`cy.fit(neighborNodes.union(selectedNode), 60);`), insert:

```typescript
      // Auto adaptive layout for focus mode: rearrange visible nodes optimally
      if (neighborNodes.length > 0) {
        const visibleCount = neighborNodes.length + 1 // neighbor nodes + selected node
        const w = containerWidth || containerRef.current?.clientWidth || 1200
        const h = containerHeight || containerRef.current?.clientHeight || 800
        const params = calcAdaptiveLayoutParams(w, h, visibleCount)
        try {
          const layout = cy.layout({
            name: 'fcose',
            quality: 'proof',
            animate: true,
            animationDuration: 400,
            nodeRepulsion: params.nodeRepulsion,
            idealEdgeLength: params.idealEdgeLength,
            gravity: params.gravity,
            numIter: params.numIter,
            tile: true,
            padding: params.padding,
          } as cytoscape.LayoutOptions)
          layout.one('layoutstop', () => {
            cy.fit(neighborNodes.union(selectedNode), params.padding)
            setZoomLevel(cy.zoom())
          })
          layout.run()
        } catch (e) {
          console.warn('[KnowledgeGraph] focus auto-layout failed:', e)
          // Fallback: cy.fit() was already called above, so we're fine
        }
      }
```

The block goes **inside** the `if (selectedConcept && focusEnabled)` branch (the focus mode branch), **after** the existing style changes and `cy.fit()` call.

- [ ] **Step 3: Verify with lsp_diagnostics**

Run: `lsp_diagnostics` on `src/components/KnowledgeGraph.tsx`

Expected: No type errors in the changed file. Pre-existing errors in server.ts / other files are unrelated.

- [ ] **Step 4: Build verification**

Run: `npm run build`

Expected: Build succeeds (exit code 0).

- [ ] **Step 5: Commit**

```bash
git add src/components/KnowledgeGraph.tsx
git commit -m "feat: auto adaptive layout when entering focus view"
```
