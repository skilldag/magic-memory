# 图谱分析面板导航 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 点击分析面板中的入口概念和路径，使左侧图谱聚焦到对应概念/路径，右侧展示概念详情。

**Architecture:** AnalysisPanel 新增 `onNavigate`/`onPathFocus` 回调 props；KnowledgeGraph 新增 `focusedNodeIds` prop 支持路径精确聚焦模式；KnowledgeGraphView 串联回调 + 状态管理。

**Tech Stack:** React 19, TypeScript, Cytoscape.js, Zustand

---

### Task 1: AnalysisPanel — 可交互的入口和路径列表

**Files:**
- Modify: `src/components/AnalysisPanel.tsx`

- [ ] **Step 1: 新增 props 接口**

将函数签名从：
```tsx
export function AnalysisPanel() {
```
改为：
```tsx
interface AnalysisPanelProps {
  onNavigate?: (conceptId: string) => void
  onPathFocus?: (conceptIds: string[]) => void
}

export function AnalysisPanel({ onNavigate, onPathFocus }: AnalysisPanelProps) {
```

- [ ] **Step 2: 入口项添加点击导航**

将入口项从第 107-115 行的纯 `<div>` 改为可点击按钮：

```tsx
{data.rootConcepts.map(r => (
  <button
    key={r.id}
    onClick={() => onNavigate?.(r.id)}
    className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md bg-blue-50/60 border border-blue-100 hover:bg-blue-100 transition-colors text-left cursor-pointer"
    title="点击聚焦到图谱"
  >
    <span className="w-2 h-2 rounded-full bg-blue-500 shrink-0" />
    <div className="flex-1 min-w-0">
      <div className="text-xs font-medium text-gray-800 truncate">{r.title}</div>
      <div className="text-[10px] text-gray-400">L{r.level} · {r.category} · 出度 {r.outDegree}</div>
    </div>
  </button>
))}
```

关键变更：`<div>` → `<button>`，新增 `onClick={() => onNavigate?.(r.id)}`，新增 `hover:bg-blue-100`、`cursor-pointer`，新增 `title`。

- [ ] **Step 3: 枢纽节点添加点击导航**

将第 130-136 行的枢纽项改为可点击：

```tsx
{data.hubConcepts.slice(0, 5).map((h, i) => (
  <button
    key={h.id}
    onClick={() => onNavigate?.(h.id)}
    className="w-full flex items-center gap-2 px-2 py-1 text-xs hover:bg-gray-50 rounded transition-colors text-left cursor-pointer"
    title="点击聚焦到图谱"
  >
    <span className="text-gray-300 font-mono w-4 shrink-0 text-right">{i + 1}</span>
    <div className="flex-1 min-w-0 truncate font-medium text-gray-700">{h.title}</div>
    <span className="text-gray-400 shrink-0">{h.totalDegree}</span>
  </button>
))}
```

关键变更：`<div>` → `<button>`，新增 `onClick`，新增 `hover:bg-gray-50 rounded cursor-pointer`。

- [ ] **Step 4: 路径项添加聚焦导航**

将路径项（第 148-165 行）改为：点击路径名区域触发 `onPathFocus`，展开折叠仍由箭头按钮独立控制。

```tsx
{data.longestPaths.slice(0, 3).map((p, i) => (
  <div key={i}>
    <div className="flex items-center w-full">
      <button
        onClick={() => setExpandedPath(expandedPath === i ? null : i)}
        className="shrink-0 flex items-center justify-center w-6 h-6 hover:bg-gray-100 rounded transition-colors"
        title="展开路径详情"
      >
        <svg width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor"
          className={`text-gray-300 transition-transform ${expandedPath === i ? 'rotate-90' : ''}`}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5l7 7-7 7" />
        </svg>
      </button>
      <button
        onClick={() => onPathFocus?.(p.pathIds)}
        className="flex-1 flex items-center gap-2 px-1 py-1.5 rounded-md hover:bg-gray-50 text-left transition-colors cursor-pointer"
        title="点击在图谱聚焦路径"
      >
        <span className="text-xs font-medium text-gray-600">{p.name} ({p.length}步)</span>
      </button>
    </div>
    {expandedPath === i && (
      <div className="px-3 pb-2">
        <PathPreview titles={p.pathTitles} />
      </div>
    )}
  </div>
))}
```

关键变更：将原本单行 `<button>` 拆分为两列——左侧独立箭头展开按钮 + 右侧可点击路径名区域。右侧按钮调用 `onPathFocus?.(p.pathIds)`。

---

### Task 2: KnowledgeGraph — 精确聚焦模式（focusedNodeIds）

**Files:**
- Modify: `src/components/KnowledgeGraph.tsx`

- [ ] **Step 1: 新增 focusedNodeIds prop**

在 `KnowledgeGraphProps` 接口（第 10-27 行）末尾添加：

```tsx
  focusedNodeIds?: string[]
```

- [ ] **Step 2: 在 props 解构中添加新参数**

在第 29-27 行的解构中添加 `focusedNodeIds`：

```tsx
export function KnowledgeGraph({
  // ... existing
  focusedNodeIds,
}: KnowledgeGraphProps) {
```

- [ ] **Step 3: 在聚焦 effect 中新增精确聚焦模式分支**

替换第 402-475 行的 effect，在 `selectedConcept` 判断之前插入 `focusedNodeIds` 分支：

```tsx
useEffect(() => {
  const cy = cyRef.current
  if (!cy) return

  const isFocusedNow = Boolean(selectedConcept && focusEnabled)

  // --- 模式 1: focusedNodeIds 精确聚焦（路径模式）---
  if (focusedNodeIds && focusedNodeIds.length > 0) {
    const focusSet = new Set(focusedNodeIds)
    const firstId = focusedNodeIds[0]

    cy.nodes().forEach(n => {
      const id = n.id()
      const isInFocus = focusSet.has(id)
      const isFirst = id === firstId
      n.style({
        'display': isInFocus ? 'element' : 'none',
        'width': isFirst ? 72 : 60,
        'height': isFirst ? 72 : 60,
        'border-width': isFirst ? 5 : 3,
        'border-color': isFirst ? '#f59e0b' : '#fff',
        'background-color': isFirst ? '#f59e0b' : (LEVEL_COLORS[Number(n.data('level'))] || '#3b82f6'),
        'opacity': 1,
      })
    })

    cy.edges().forEach(e => {
      const srcInFocus = focusSet.has(e.data('source'))
      const tgtInFocus = focusSet.has(e.data('target'))
      const show = srcInFocus && tgtInFocus
      e.style({
        'display': show ? 'element' : 'none',
        'opacity': show ? 1 : 0,
        'width': show ? 2 : 0.5,
      })
    })

    // 聚焦到整个路径节点
    const focusNodes = cy.nodes().filter(n => focusSet.has(n.id()))
    if (focusNodes.length > 0) {
      cy.fit(focusNodes, 60)
    }
    wasFocusedRef.current = true
    return
  }

  // --- 模式 2: 原有逻辑（复位/单概念 auto-neighbor）---
  if (!selectedConcept || !focusEnabled) {
    // ... existing reset logic (lines 408-437) unchanged ...
    cy.nodes().forEach(n => {
      const isSelected = selectedConcept && n.id() === selectedConcept.id
      n.style({
        'display': 'element',
        'width': 50,
        'height': 50,
        'border-width': isSelected ? 4 : 2,
        'border-color': isSelected ? '#f59e0b' : '#fff',
        'background-color': isSelected ? '#f59e0b' : (LEVEL_COLORS[Number(n.data('level'))] || '#3b82f6'),
        'opacity': 1,
      })
    })

    cy.edges().forEach(e => {
      const sourceSelected = selectedConcept && e.data('source') === selectedConcept.id
      const targetSelected = selectedConcept && e.data('target') === selectedConcept.id
      e.style({
        'display': 'element',
        'opacity': !selectedConcept || sourceSelected || targetSelected ? 1 : 0.15,
        'width': !selectedConcept || sourceSelected || targetSelected ? 3 : 1,
      })
    })

    if (wasFocusedRef.current) {
      cy.fit(undefined, 50)
      setZoomLevel(cy.zoom())
    }
    wasFocusedRef.current = false
    return
  }

  // --- 模式 3: 单概念 auto-neighbor 聚焦（已有逻辑 lines 440-474）---
  const selectedNode = cy.getElementById(selectedConcept.id)
  const connectedEdges = selectedNode.connectedEdges()
  const relevantEdges = connectedEdges.filter(e => e.data('edgeType') !== 'related')
  const neighborNodes = relevantEdges.connectedNodes()
  const relatedNodeIds = new Set([selectedConcept.id, ...neighborNodes.map(n => n.id())])

  cy.nodes().forEach(n => {
    const isSelected = n.id() === selectedConcept.id
    const isRelated = relatedNodeIds.has(n.id())
    n.style({
      'display': isRelated ? 'element' : 'none',
      'width': isSelected ? 72 : 60,
      'height': isSelected ? 72 : 60,
      'border-width': isSelected ? 5 : 3,
      'border-color': isSelected ? '#f59e0b' : '#fff',
      'background-color': isSelected ? '#f59e0b' : (LEVEL_COLORS[Number(n.data('level'))] || '#3b82f6'),
      'opacity': 1,
    })
  })

  cy.edges().forEach(e => {
    const sourceVisible = relatedNodeIds.has(e.data('source'))
    const targetVisible = relatedNodeIds.has(e.data('target'))
    const isVisible = sourceVisible && targetVisible
    const touchesSelected = e.data('source') === selectedConcept.id || e.data('target') === selectedConcept.id
    e.style({
      'display': isVisible ? 'element' : 'none',
      'opacity': isVisible ? 1 : 0,
      'width': touchesSelected ? 4 : 2,
    })
  })

  cy.fit(neighborNodes.union(selectedNode), 60)
  wasFocusedRef.current = isFocusedNow
}, [selectedConcept, focusEnabled, focusedNodeIds])
```

**注意**: `useEffect` 依赖数组末尾添加 `focusedNodeIds`。

- [ ] **Step 4: 清理旧 props 联动**

检查 `focusedNodeIds` 是否需要在其他 effect 中清除。当前 `handleSelectConcept` 在 KnowledgeGraphView 中重置状态即可，KnowledgeGraph 内部不需要额外清除逻辑。

---

### Task 3: KnowledgeGraphView — 串联回调与状态

**Files:**
- Modify: `src/components/KnowledgeGraphView.tsx`

- [ ] **Step 1: 新增 focusedNodeIds 状态**

在第 35 行附近（`selectedConceptId` state 之后）添加：

```tsx
const [focusedNodeIds, setFocusedNodeIds] = useState<string[] | undefined>(undefined)
```

- [ ] **Step 2: handleSelectConcept 重置 focusedNodeIds**

在 `handleSelectConcept`（第 127 行）中添加：

```tsx
const handleSelectConcept = (concept: Concept) => {
  selectedConceptRef.current = concept
  setSelectedConceptId(concept.id)
  selectConcept(concept)
  cancelHideHoverActions()
  setHoverConcept(null)
  setFocusedNodeIds(undefined)  // 重置路径聚焦
}
```

- [ ] **Step 3: handleNavigate 重置 focusedNodeIds**

在 `handleNavigate`（第 163 行）中添加：

```tsx
const handleNavigate = (conceptId: string) => {
  const concept = concepts.find(c => c.id === conceptId)
  if (concept) {
    setSelectedConceptId(concept.id)
    selectedConceptRef.current = concept
    setFocusedNodeIds(undefined)
  }
}
```

- [ ] **Step 4: handlePathFocus 回调**

添加新函数（在 `handleNavigate` 之后）：

```tsx
const handlePathFocus = useCallback((ids: string[]) => {
  if (ids.length === 0) return
  setFocusedNodeIds(ids)
  // 导航到第一个概念（右侧展示详情）
  const first = concepts.find(c => c.id === ids[0])
  if (first) {
    setSelectedConceptId(first.id)
    selectedConceptRef.current = first
    selectConcept(first)
  }
}, [concepts, selectConcept])
```

- [ ] **Step 5: 传递给 KnowledgeGraph**

在 `<KnowledgeGraph>` 调用（第 455-477 行）中添加：

```tsx
<KnowledgeGraph
  // ... existing props
  focusedNodeIds={focusedNodeIds}
/>
```

- [ ] **Step 6: 传递给 AnalysisPanel**

在第 501-512 行将 `<AnalysisPanel />` 改为：

```tsx
<AnalysisPanel
  onNavigate={handleNavigate}
  onPathFocus={handlePathFocus}
/>
```

- [ ] **Step 7: 编译检查**

运行类型检查确保无编译错误：

```bash
npm run typecheck
```

预期：通过无报错。

- [ ] **Step 8: 提交**

```bash
git add src/components/AnalysisPanel.tsx src/components/KnowledgeGraph.tsx src/components/KnowledgeGraphView.tsx
git commit -m "feat: make analysis panel entries and paths clickable for graph focus"
```
