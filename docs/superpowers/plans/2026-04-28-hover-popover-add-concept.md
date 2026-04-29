# Hover Popover Add Concept Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed top-right action bar with a floating popover near the hovered Cytoscape node, providing AI generation and manual concept creation.

**Architecture:** Single-file change to `KnowledgeGraphView.tsx` — add an inline `ConceptHoverPopover` component that uses existing `x,y` from Cytoscape's `renderedPosition()` to position absolutely within the graph container. Reuses all existing dialog states and handlers.

**Tech Stack:** React 19, TypeScript, Tailwind CSS 4, Cytoscape.js, Zustand

**Files Modified:**
- `magic-memory/src/components/KnowledgeGraphView.tsx` (only file changed)

---

### Task 1: Add ConceptHoverPopover component

**File:** Modify `magic-memory/src/components/KnowledgeGraphView.tsx`

Add the inline popover component inside the same file, before the `KnowledgeGraphView` function.

- [ ] **Step 1: Add ConceptHoverPopover type and component**

Insert this code before line 8 (before `interface KnowledgeGraphViewProps`):

```typescript
interface ConceptHoverPopoverProps {
  concept: Concept
  x: number
  y: number
  containerWidth: number
  containerHeight: number
  onExplore: (concept: Concept) => void
  onManualAdd: (concept: Concept) => void
  onClose: () => void
}

function ConceptHoverPopover({ concept, x, y, containerWidth, containerHeight, onExplore, onManualAdd, onClose }: ConceptHoverPopoverProps) {
  const POPOVER_WIDTH = 200
  const POPOVER_HEIGHT = 120
  const GAP = 12

  // 默认右侧，超出边界切换到左侧
  const anchorRight = x + GAP + POPOVER_WIDTH < containerWidth
  const left = anchorRight ? x + GAP : x - GAP - POPOVER_WIDTH

  // 垂直居中，超出边界则偏移
  let top = y - POPOVER_HEIGHT / 2
  if (top < 8) top = 8
  if (top + POPOVER_HEIGHT > containerHeight - 8) top = containerHeight - POPOVER_HEIGHT - 8

  // 如果容器太小放不下浮层，回退到右上角
  const fallback = POPOVER_WIDTH + GAP * 2 > containerWidth
  const finalLeft = fallback ? containerWidth - POPOVER_WIDTH - 12 : left
  const finalTop = fallback ? 12 : top

  return (
    <div
      className="absolute z-30"
      style={{ left: finalLeft, top: finalTop }}
      onMouseEnter={e => { e.stopPropagation() }}
    >
      <div className="bg-white rounded-xl shadow-xl border border-gray-200 overflow-hidden" style={{ width: POPOVER_WIDTH }}>
        <div className="px-3.5 py-2.5 border-b border-gray-100">
          <div className="text-xs font-medium text-gray-400 uppercase tracking-wider">延伸</div>
          <div className="text-sm font-semibold text-gray-900 truncate mt-0.5" title={concept.title}>
            {concept.title}
          </div>
        </div>
        <div className="p-2.5 space-y-1.5">
          <button
            type="button"
            onMouseDown={e => e.preventDefault()}
            onClick={() => onExplore(concept)}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 text-sm font-medium transition-colors"
          >
            <span>🤖</span>
            <span>AI 生成探索</span>
          </button>
          <button
            type="button"
            onMouseDown={e => e.preventDefault()}
            onClick={() => onManualAdd(concept)}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 text-gray-700 hover:bg-gray-100 text-sm font-medium transition-colors"
          >
            <span>✏️</span>
            <span>手动添加概念</span>
          </button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify insertion is valid**

Run: `npx tsc --noEmit --pretty 2>&1 | head -30`
Expected: No type errors from the new code.

---

### Task 2: Replace fixed action bar with popover

**File:** Modify `magic-memory/src/components/KnowledgeGraphView.tsx`

- [ ] **Step 1: Add container ref and dimension tracking state**

Find the `KnowledgeGraphView` function (around line 12), add these state declarations after the existing state (around line 38):

```typescript
const graphContainerRef = useRef<HTMLDivElement>(null)
const [containerSize, setContainerSize] = useState({ width: 0, height: 0 })
```

Also need to add `useRef` to the imports. The current import (line 1) is:
```typescript
import { useEffect, useMemo, useState } from 'react'
```

Change to:
```typescript
import { useEffect, useMemo, useState, useRef } from 'react'
```

- [ ] **Step 2: Add ResizeObserver to track container dimensions**

Insert after the `useEffect` for `loadGraph` (after line 43):

```typescript
useEffect(() => {
  const el = graphContainerRef.current
  if (!el) return
  const ro = new ResizeObserver(entries => {
    for (const entry of entries) {
      const { width, height } = entry.contentRect
      setContainerSize({ width, height })
    }
  })
  ro.observe(el)
  // 初始值
  setContainerSize({ width: el.clientWidth, height: el.clientHeight })
  return () => ro.disconnect()
}, [])
```

- [ ] **Step 3: Add ref to the graph container div**

Find the graph container div (the `flex-1 min-w-0 relative` one, around line 317). Add `ref={graphContainerRef}`:

```typescript
<div ref={graphContainerRef} className="flex-1 min-w-0 relative">
```

- [ ] **Step 4: Update onHoverConcept handler to pass coordinates**

Find the `onHoverConcept` handler (around line 342):

```typescript
onHoverConcept={payload => {
  if (viewMode !== 'explore' || !selectedConcept) return
  cancelHideHoverActions()
  setActiveHoverAction(null)
  setHoverConcept(payload)
}}
```

Change to:
```typescript
onHoverConcept={payload => {
  if (viewMode !== 'explore' || !selectedConcept) return
  cancelHideHoverActions()
  setActiveHoverAction(null)
  setHoverConcept(payload)
  // 容器尺寸由 ResizeObserver 自动更新
}}
```

- [ ] **Step 5: Delete the old fixed action bar**

Find and **delete** this entire block (around lines 351-398):

```typescript
{viewMode === 'explore' && actionTargetConcept && (
  <div
    className="absolute top-3 right-16 z-20 flex items-center gap-2 bg-white/95 backdrop-blur border border-gray-200 rounded-lg shadow px-2 py-1.5"
    ...
  >
    ...
  </div>
)}
```

- [ ] **Step 6: Add popover rendering in place of deleted bar**

At the same position where the old bar was (inside the `flex-1 min-w-0 relative` div, after the KnowledgeGraph component but still inside the container), add:

```typescript
{hoverConcept && viewMode === 'explore' && selectedConcept && (
  <ConceptHoverPopover
    concept={hoverConcept.concept}
    x={hoverConcept.x}
    y={hoverConcept.y}
    containerWidth={containerSize.width}
    containerHeight={containerSize.height}
    onExplore={concept => {
      setActionConcept(concept)
      setShowExploreDialog(true)
    }}
    onManualAdd={concept => {
      setActionConcept(concept)
      setShowManualLinkDialog(true)
    }}
    onClose={() => {
      setHoverConcept(null)
    }}
  />
)}
```

Note: The existing `showExploreDialog` rendering at line 808 already handles the explore dialog. But it currently checks `selectedConcept` — we need to make sure the ExploreDialog can also be opened from the popover for any hovered concept. Let me check line 808:

```typescript
{showExploreDialog && selectedConcept && (
  <ExploreDialog
    sourceConcept={selectedConcept}
    onClose={() => setShowExploreDialog(false)}
  />
)}
```

This passes `selectedConcept` as the source. But when triggered from the popover, the concept might be different from `selectedConcept`. We need to change this to use `actionConcept` instead. Update line 808 to:

```typescript
{showExploreDialog && actionConcept && (
  <ExploreDialog
    sourceConcept={actionConcept}
    onClose={() => {
      setShowExploreDialog(false)
      setActiveHoverAction(null)
      setHoverConcept(null)
    }}
  />
)}
```

- [ ] **Step 7: Update manual link dialog and batch link dialog source concept**

Find the `showManualLinkDialog && actionConcept &&` dialog (around line 814) and verify it already uses `actionConcept` — yes it does, so no change needed.

Find the `showBatchLinkDialog && actionConcept &&` dialog (around line 843) and verify it already uses `actionConcept` — yes it does, so no change needed.

- [ ] **Step 8: Check for stale state issues**

The existing `scheduleHideHoverActions` and `cancelHideHoverActions` already handle the delayed hiding. Ensure the popover's `onMouseEnter` calls `cancelHideHoverActions()` so it doesn't disappear while the user interacts with it.

Add to the KnowledgeGraphView's JSX, alongside the popover rendering:

```typescript
{/* 透明覆盖层：将鼠标事件桥接到浮层的 enter/leave */}
{hoverConcept && viewMode === 'explore' && selectedConcept && (
  <>
    <ConceptHoverPopover ... />
    {/* 原有鼠标移入浮层保持显示的逻辑已由 popover 内部 onMouseEnter 处理 */}
  </>
)}
```

The existing `scheduleHideHoverActions` at line 297-312 and `cancelHideHoverActions` at line 304 are already sufficient. The popover's `onMouseEnter` calls `e.stopPropagation()` which prevents the event from reaching the graph container, and the `onHoverLeave` will fire when the mouse leaves the node. We don't need additional bridge logic.

---

### Task 3: Cleanup stale states and edge cases

- [ ] **Step 1: Clean up unused state variables**

After removing the fixed action bar, check if these state variables are still needed:
- `actionConcept`: Still used by `handleManualAdd`, `generateBatchSuggestions`, `confirmBatchAdd`, and both dialogs → **keep**
- `activeHoverAction`: Was used to toggle between 'manual' and 'batch' in the old bar. Still used in the dialogs for cleanup → **keep**
- `isHoverActionsActive`: Only used by `onMouseEnter`/`onMouseLeave` on the old bar → **can be removed** (no longer needed since popover handles its own event propagation)

Find and remove `isHoverActionsActive`:
1. Remove from state declarations (line 30)
2. Remove its usage in `onMouseEnter` handler (line 355-359) — already deleting that block

- [ ] **Step 2: Verify the hover concept is cleared when leaving the graph area**

The existing `onHoverLeave` handler on the KnowledgeGraph component fires `scheduleHideHoverActions()` which sets a 300ms timer to clear `hoverConcept`. This is correct.

Add one more safeguard: if the user clicks a dialog button, immediately clear the hover state:

In the `onExplore` handler (passed to popover), add `setHoverConcept(null)`:
```typescript
onExplore={concept => {
  setActionConcept(concept)
  setShowExploreDialog(true)
  setHoverConcept(null)  // 关闭浮层
}}
```

Same for `onManualAdd`:
```typescript
onManualAdd={concept => {
  setActionConcept(concept)
  setShowManualLinkDialog(true)
  setHoverConcept(null)  // 关闭浮层
}}
```

---

### Task 4: Verification

- [ ] **Step 1: Clean up unused `isHoverActionsActive`**

After removing the state, also remove its references in `cancelHideHoverActions` (line 298) and `scheduleHideHoverActions` (line 307):

```typescript
// Remove: && !isHoverActionsActive
// Change from:
if (!showManualLinkDialog && !showBatchLinkDialog && !isHoverActionsActive) {
// To:
if (!showManualLinkDialog && !showBatchLinkDialog) {
```

- [ ] **Step 2: LSP diagnostics check**

Run: `npx tsc --noEmit --pretty`
Expected: 0 errors, 0 warnings

- [ ] **Step 3: Build check**

Run: `npm run build`
Expected: Build succeeds

- [ ] **Step 4: Manual test**

1. Start dev server: `npm run dev`
2. Open browser to the app
3. Click a concept node (ensure explore mode)
4. Hover over another node in the graph
5. Verify: popover appears near the hovered node, not at top-right
6. Click "AI 生成探索" → ExploreDialog opens
7. Close dialog
8. Hover again → popover appears
9. Click "手动添加概念" → manual dialog opens
10. Move mouse away from node and popover → popover disappears after 300ms
