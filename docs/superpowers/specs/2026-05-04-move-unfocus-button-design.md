# Move Unfocus Button to Knowledge Graph Area

**Date**: 2026-05-04
**Status**: Approved
**Author**: Sisyphus

## Problem

The "取消聚焦视图" (unfocus/cancel focus) button is currently located in the right-side concept detail panel (`ConceptDetailPanel.tsx`). When a concept is selected, the knowledge graph enters focus mode. To exit focus mode, the user must click the × button in the right panel — which is visually distant from the graph itself.

The user wants this button moved to the left-side knowledge graph area for faster access and better UX.

## Current Architecture

### Layout Structure

```
┌──────────────────────────┬──────────┬────────────────────────────┐
│  知识图谱区域               │ 拖拽分割线 │  右侧概念详情面板            │
│  (KnowledgeGraphView)    │  (40px)  │  (ConceptDetailPanel)     │
│                          │          │                            │
│  ┌─ 图谱 ────────────┐   │          │  ┌─ 标题 ──── ✕ ──────┐   │
│  │                   │   │          │  │  (取消选择)         │   │
│  │                   │   │          │  │                     │   │
│  │    知识图谱        │   │          │  │  查阅文档/导入/...  │   │
│  │                   │   │          │  │                     │   │
│  └───────────────────┘   │          │  └─────────────────────┘   │
└──────────────────────────┴──────────┴────────────────────────────┘
```

### Current Focus Flow

1. User clicks a concept node in KnowledgeGraph
2. `KnowledgeGraphView.handleSelectConcept()` sets `selectedConceptId` and calls `selectConcept()`
3. `KnowledgeGraph` detects `selectedConcept` + `focusEnabled=true` → enters focus mode (shows only selected node + neighbors)
4. Right panel shows `ConceptDetailPanel` with the concept details
5. User clicks × button → `onDeselect` → `setSelectedConceptId(null)` → graph restores to full view

### Key Files

| File | Role |
|------|------|
| `src/components/KnowledgeGraphView.tsx` | Main layout: left=graph, right=panel. Manages `selectedConceptId` state |
| `src/components/ConceptDetailPanel.tsx` | Right panel. Has × button (line 281) calling `onDeselect` |
| `src/components/KnowledgeGraph.tsx` | Graph visualization with focus logic |

## Design

### Floating Top Bar on Graph

When the graph is in focus mode (a concept is selected), show a floating bar at the top of the graph area. This bar provides:
- Visual feedback that focus mode is active
- A button to exit focus mode

```
┌────────────────────────────────────────────────┐
│  ◉ 聚焦: PagedAttention              [退出聚焦] │ ← Floating bar (only when focused)
│                                                │
│           ┌───┐     ┌───┐                      │
│           │ A │────▶│ B │                      │
│           └───┘     └───┘                      │
│                       │                        │
│           ┌───┐     ┌───┐                      │
│           │ C │◀────│ D │                      │
│           └───┘     └───┘                      │
└────────────────────────────────────────────────┘
```

### Specifications

**Position**: Absolute positioned at top of the graph container (`graphContainerRef` div), full width, z-index above the graph

**Visibility**: Only visible when `selectedConcept !== null` and NOT in `processMode`

**Content**:
- Left: Green dot icon + "聚焦: {concept.title}" text
- Right: "退出聚焦" button (or × icon button)

**Click behavior**: Calls same deselect logic as current × button:
```typescript
setSelectedConceptId(null);
useKnowledgeGraphStore.setState({ selectedConcept: null });
```

**Style**:
- Semi-transparent background (backdrop-blur/glass effect)
- Padding: px-4 py-2
- Does not block graph interactions (pointer-events: auto only on the button, or the bar can be narrow)

### Changes to Right Panel

Remove the × button from `ConceptDetailPanel.tsx` (line 281-285) since the same functionality is now available from the graph area. The right panel becomes cleaner without the redundant close button.

### Boundary / States

| State | Left Graph Area | Right Panel |
|-------|----------------|-------------|
| No concept selected | Graph shows full view. **No floating bar** | SummaryPanel |
| Concept selected (focus) | Graph shows focused nodes. **Floating bar visible** | ConceptDetailPanel (no ×) |
| ProcessCanvas mode | ProcessCanvas shown. **No floating bar** | ConceptDetailPanel (no ×) |

## Implementation Summary

### Files to Modify

1. **`KnowledgeGraphView.tsx`**: Add floating bar JSX inside `graphContainerRef` div, conditional on `selectedConcept` state
2. **`ConceptDetailPanel.tsx`**: Remove the × button (lines 280-285)

### Minimal Change

This is a pure UI repositioning — no logic changes, no state changes, no new props. The `onDeselect` callback already exists and is passed correctly from `KnowledgeGraphView` through to `ConceptDetailPanel`.

## Acceptance Criteria

- [ ] When no concept is selected, no floating bar is shown
- [ ] When a concept is selected (focus mode), a floating bar appears at the top of the graph
- [ ] The bar shows the focused concept name
- [ ] Clicking "退出聚焦" exits focus mode and restores full graph view
- [ ] The × button is removed from the right panel
- [ ] Graph interactions (zoom, pan, node click) still work with the bar present
- [ ] ProcessCanvas mode does NOT show the bar
