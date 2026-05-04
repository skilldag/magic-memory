# Smart Layout Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Smart Fit" button to the knowledge graph that dynamically recalculates fcose layout parameters based on container size and node count, then re-laysout the graph to perfectly fill the viewport.

**Architecture:** Add a pure function `calcAdaptiveLayoutParams()` to compute layout parameters, a `handleSmartLayout()` callback in KnowledgeGraph that re-runs fcose with those params, a button UI element in the existing zoom controls panel, and pass `containerWidth`/`containerHeight` props from KnowledgeGraphView (which already tracks container size via `useContainerSize`).

**Tech Stack:** React 19, TypeScript, Cytoscape.js + fcose

---

### Task 1: Write test for calcAdaptiveLayoutParams utility

**Files:**
- Create: `tests/adaptiveLayout.test.ts`

- [ ] **Step 1: Write the test file**

```typescript
// tests/adaptiveLayout.test.ts

// Mirror the exact function we'll implement in KnowledgeGraph.tsx
interface AdaptiveParams {
  idealEdgeLength: number
  nodeRepulsion: number
  gravity: number
  padding: number
  numIter: number
}

function calcAdaptiveLayoutParams(
  containerWidth: number,
  containerHeight: number,
  nodeCount: number
): AdaptiveParams {
  const diagonal = Math.sqrt(containerWidth * containerWidth + containerHeight * containerHeight)
  const idealEdgeLength = Math.max(60, Math.min(200, (diagonal / Math.sqrt(Math.max(nodeCount, 1))) * 1.2))
  const nodeRepulsion = Math.max(8000, Math.min(50000, idealEdgeLength * nodeCount * 3))
  const gravity = Math.max(0.02, Math.min(0.3, 100 / (nodeCount + 10)))
  const padding = Math.max(20, Math.min(80, Math.min(containerWidth, containerHeight) * 0.06))
  const numIter = Math.max(100, Math.min(800, nodeCount * 8))
  return { idealEdgeLength, nodeRepulsion, gravity, padding, numIter }
}

async function main() {
  let passed = 0
  let failed = 0

  function ok(condition: boolean, msg: string) {
    if (condition) { console.log(`  ok ${msg}`); passed++ }
    else { console.log(`  FAIL ${msg}`); failed++ }
  }

  // Test 1: normal case — 20 nodes in 1200x800 container
  const r1 = calcAdaptiveLayoutParams(1200, 800, 20)
  ok(r1.idealEdgeLength >= 60 && r1.idealEdgeLength <= 200, 'idealEdgeLength in range')
  ok(r1.nodeRepulsion >= 8000 && r1.nodeRepulsion <= 50000, 'nodeRepulsion in range')
  ok(r1.gravity >= 0.02 && r1.gravity <= 0.3, 'gravity in range')
  ok(r1.padding >= 20 && r1.padding <= 80, 'padding in range')
  ok(r1.numIter >= 100 && r1.numIter <= 800, 'numIter in range')

  // Test 2: small container — 5 nodes in 400x300
  const r2 = calcAdaptiveLayoutParams(400, 300, 5)
  ok(r2.idealEdgeLength >= 60, 'small container: edge length min clamped')
  ok(r2.padding >= 20, 'small container: padding min clamped')

  // Test 3: large graph — 100 nodes in 1920x1080
  const r3 = calcAdaptiveLayoutParams(1920, 1080, 100)
  ok(r3.idealEdgeLength <= 200, 'large graph: edge length max clamped')
  ok(r3.nodeRepulsion >= 8000, 'large graph: repulsion scales up')
  ok(r3.numIter <= 800, 'large graph: numIter max clamped')

  // Test 4: single node — 800x600, 1 node
  const r4 = calcAdaptiveLayoutParams(800, 600, 1)
  ok(r4.numIter >= 100, 'single node: min iter')
  ok(r4.gravity >= 0.02, 'single node: gravity min clamped')

  console.log(`\n${passed} passed, ${failed} failed`)
  if (failed > 0) process.exit(1)
}

main()
```

- [ ] **Step 2: Run test — it should pass (function is self-contained)**

Run: `bun run tests/adaptiveLayout.test.ts`
Expected: `5 passed, 0 failed`

- [ ] **Step 3: Commit**

```bash
git add tests/adaptiveLayout.test.ts
git commit -m "test: add calcAdaptiveLayoutParams unit test"
```

---

### Task 2: Add containerWidth/containerHeight props to KnowledgeGraph

**Files:**
- Modify: `src/components/KnowledgeGraph.tsx:10-29` (props interface)
- Modify: `src/components/KnowledgeGraph.tsx:31-49` (component function signature)
- Modify: `src/components/KnowledgeGraphView.tsx:398-421` (pass props)

- [ ] **Step 1: Add props to KnowledgeGraphProps interface**

In `src/components/KnowledgeGraph.tsx`, add to `KnowledgeGraphProps` (after line 28):

```typescript
  // Container dimensions for adaptive layout
  containerWidth?: number
  containerHeight?: number
```

- [ ] **Step 2: Destructure new props in function signature**

In `src/components/KnowledgeGraph.tsx`, add `containerWidth, containerHeight` to the destructured props (around line 31-49). Insert after `focusedNodeIds`:

```typescript
  containerWidth = 1200,
  containerHeight = 800,
```

- [ ] **Step 3: Pass containerSize from KnowledgeGraphView**

In `src/components/KnowledgeGraphView.tsx`, add `containerWidth` and `containerHeight` to the `<KnowledgeGraph>` JSX (around line 398-421). Insert after `focusedNodeIds={focusedNodeIds}`:

```typescript
            containerWidth={containerSize?.width}
            containerHeight={containerSize?.height}
```

- [ ] **Step 4: Verify typecheck**

Run: `npm run typecheck`
Expected: No type errors

- [ ] **Step 5: Commit**

```bash
git add src/components/KnowledgeGraph.tsx src/components/KnowledgeGraphView.tsx
git commit -m "feat: pass container dimensions to KnowledgeGraph for adaptive layout"
```

---

### Task 3: Add calcAdaptiveLayoutParams function + handleSmartLayout callback

**Files:**
- Modify: `src/components/KnowledgeGraph.tsx` (add utility function + callback)

- [ ] **Step 1: Add calcAdaptiveLayoutParams before the component function**

Insert the utility function after the imports (line 6-8) and before the `KnowledgeGraphProps` interface (line 10):

```typescript
interface AdaptiveParams {
  idealEdgeLength: number
  nodeRepulsion: number
  gravity: number
  padding: number
  numIter: number
}

function calcAdaptiveLayoutParams(
  containerWidth: number,
  containerHeight: number,
  nodeCount: number
): AdaptiveParams {
  const diagonal = Math.sqrt(containerWidth * containerWidth + containerHeight * containerHeight)
  const idealEdgeLength = Math.max(60, Math.min(200, (diagonal / Math.sqrt(Math.max(nodeCount, 1))) * 1.2))
  const nodeRepulsion = Math.max(8000, Math.min(50000, idealEdgeLength * nodeCount * 3))
  const gravity = Math.max(0.02, Math.min(0.3, 100 / (nodeCount + 10)))
  const padding = Math.max(20, Math.min(80, Math.min(containerWidth, containerHeight) * 0.06))
  const numIter = Math.max(100, Math.min(800, nodeCount * 8))
  return { idealEdgeLength, nodeRepulsion, gravity, padding, numIter }
}
```

- [ ] **Step 2: Add handleSmartLayout callback after handleFit**

Insert after `handleFit` (line 97) in `src/components/KnowledgeGraph.tsx`:

```typescript
  const handleSmartLayout = useCallback(() => {
    const cy = cyRef.current
    if (!cy) return
    const count = concepts.length
    if (count <= 1) {
      cy.fit(undefined, 50)
      setZoomLevel(cy.zoom())
      return
    }
    const w = containerWidth || containerRef.current?.clientWidth || 1200
    const h = containerHeight || containerRef.current?.clientHeight || 800
    const params = calcAdaptiveLayoutParams(w, h, count)
    try {
      const layout = cy.layout({
        name: 'fcose',
        quality: 'default',
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
        cy.fit(undefined, params.padding)
        setZoomLevel(cy.zoom())
      })
      layout.run()
    } catch (e) {
      console.warn('[KnowledgeGraph] smart layout failed:', e)
      // Fallback: basic fit
      cy.fit(undefined, 50)
      setZoomLevel(cy.zoom())
    }
  }, [concepts.length, containerWidth, containerHeight])
```

- [ ] **Step 3: Verify typecheck**

Run: `npm run typecheck`
Expected: No type errors

- [ ] **Step 4: Commit**

```bash
git add src/components/KnowledgeGraph.tsx
git commit -m "feat: add calcAdaptiveLayoutParams and handleSmartLayout to KnowledgeGraph"
```

---

### Task 4: Add Smart Layout button UI

**Files:**
- Modify: `src/components/KnowledgeGraph.tsx:563-590` (add button to zoom controls panel)

- [ ] **Step 1: Add the Smart Layout button after the fit button**

In `src/components/KnowledgeGraph.tsx`, after the fit button's closing `</button>` (line 587) and before the zoom level `<div>` (line 588), insert:

```typescript
            <button
              onClick={handleSmartLayout}
              className="w-8 h-8 bg-white rounded shadow flex items-center justify-center hover:bg-gray-100 text-gray-700"
              title="自适应布局"
            >
              <svg width={16} height={16} className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
              </svg>
            </button>
```

This reuses the same "fit to bounds" SVG icon as the existing fit button — appropriate since the smart layout button is an enhanced version of fit.

- [ ] **Step 2: Verify the button renders correctly in context**

The rendered zoom panel will look like (after change):

```typescript
      {isReady && (
        <div className="absolute top-3 right-3 flex flex-col gap-1">
          <button onClick={handleZoomIn} ...>+</button>
          <button onClick={handleZoomOut} ...>−</button>
          <button onClick={handleFit} ...>⟷ icon</button>
          <button onClick={handleSmartLayout} ...>⟷ icon (自适应布局)</button>
          <div className="text-center ...">80%</div>
        </div>
      )}
```

- [ ] **Step 3: Run typecheck + build**

Run: `npm run typecheck && npm run build`
Expected: Both pass with no errors

- [ ] **Step 4: Commit**

```bash
git add src/components/KnowledgeGraph.tsx
git commit -m "feat: add smart layout button to knowledge graph zoom controls"
```

---

### Task 5: Final verification

- [ ] **Step 1: Re-run adaptive layout test**

Run: `bun run tests/adaptiveLayout.test.ts`
Expected: `5 passed, 0 failed`

- [ ] **Step 2: Full typecheck**

Run: `npm run typecheck`
Expected: No type errors

- [ ] **Step 3: Full build**

Run: `npm run build`
Expected: Build succeeds

- [ ] **Step 4: Review changed files**

```bash
git diff --stat
```

Expected: Only `src/components/KnowledgeGraph.tsx`, `src/components/KnowledgeGraphView.tsx`, `tests/adaptiveLayout.test.ts` changed.
