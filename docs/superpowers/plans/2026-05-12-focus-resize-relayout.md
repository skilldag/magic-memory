# 聚焦视图拉伸后自适应布局实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 拖拽右侧面板分割线拉伸聚焦视图后，在 mouseup 时自动触发一次 fcose 自适应布局。

**Architecture:** 给 KnowledgeGraph 新增 `relayoutKey` prop，KnowledgeGraphView 在拖拽结束的 mouseup 中递增此 key，KnowledgeGraph 通过 useEffect watch key 变化调用 handleSmartLayout。

**Tech Stack:** React + TypeScript + Cytoscape.js + fcose

---

## File Structure

| 文件 | 改动类型 | 职责 |
|------|----------|------|
| `src/components/KnowledgeGraph.tsx` | 修改 | 新增 `relayoutKey?: number` prop；新增 useEffect watch key 变化调 handleSmartLayout |
| `src/components/KnowledgeGraphView.tsx` | 修改 | 新增 `relayoutKey` state；在 divider mouseup 中检测聚焦模式并递增 |

无新建文件，无测试（纯 UI 时序行为）。

---

### Task 1: KnowledgeGraph — 新增 relayoutKey prop + watch effect

**Files:**
- Modify: `src/components/KnowledgeGraph.tsx:32-58` (interface) + 新增 effect

- [ ] **Step 1: 在 Interface 中新增 relayoutKey prop**

在 `KnowledgeGraphProps` interface 末尾（`onDeleteEdge` 之后）添加：

```typescript
export function KnowledgeGraph({
  // ... 末尾添加
  relayoutKey,
}: KnowledgeGraphProps) {
```

- [ ] **Step 2: 新增 handleSmartLayout 的 ref 兜底 + watch effect**

在 `KnowledgeGraph` 组件内部，`handleSmartLayout` useCallback 之后，增量更新 effect 之前，添加：

```typescript
// ref 兜底，避免 effect 中引用陈旧的 handleSmartLayout
const handleSmartLayoutRef = useRef(handleSmartLayout)
handleSmartLayoutRef.current = handleSmartLayout

// watch relayoutKey 递增 → 触发自适应布局
useEffect(() => {
  if (relayoutKey && relayoutKey > 0) {
    handleSmartLayoutRef.current()
  }
}, [relayoutKey])
```

注意：依赖数组只包含 `relayoutKey`，不包含 `handleSmartLayout`（用 ref 避免闭包陈旧 + 不必要的 effect 重跑）。

---

### Task 2: KnowledgeGraphView — 新增 state + mouseup 触发

**Files:**
- Modify: `src/components/KnowledgeGraphView.tsx`

- [ ] **Step 1: 新增 relayoutKey state**

在 `KnowledgeGraphView` 函数内，现有 state 声明区域（约第 57 行 `focusedNodeIds` 附近）添加：

```typescript
const [relayoutKey, setRelayoutKey] = useState(0)
```

- [ ] **Step 2: 传递给 KnowledgeGraph**

找到 KnowledgeGraph JSX（约第 396 行），在现有 props 中插入：

```tsx
<KnowledgeGraph
  // ... 现有 props
  relayoutKey={selectedConcept ? relayoutKey : 0}
  // ...
/>
```

使用 `selectedConcept` 条件：非聚焦模式时传 0，effect 不会触发（watch 条件 `relayoutKey > 0`）。

- [ ] **Step 3: 在 divider mouseup 中递增**

找到 `handleDividerMouseDown` 函数中的 `handleMouseUp`（约第 108-113 行），在末尾追加：

```typescript
const handleMouseUp = () => {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
  // 聚焦模式下拖拽结束时触发图谱自适应布局
  if (selectedConceptRef.current) {
    setRelayoutKey(k => k + 1)
  }
}
```

- [ ] **Step 4: 验证编译**

```bash
cd /Users/meetai/source/magic-memory && npx tsc --noEmit
```

Expected: 无新增类型错误（已有错误为 Bun/Node types 缺失，与本改动无关）。

---

## 验证清单

| 检查项 | 方法 |
|--------|------|
| 非聚焦模式拖拽不触发 | 不选中概念，拖拽分割线 → 图谱不应重布局 |
| 聚焦模式拖拽触发 | 选中概念，拖拽分割线 → 松开后图谱自动重排 |
| 多次拖拽正常 | 连续拖拽多次，每次松开都触发一次重布局 |
| 边界：极小/极大宽度 | 拖拽到最小/最大宽度，布局参数 clamp 正常 |
