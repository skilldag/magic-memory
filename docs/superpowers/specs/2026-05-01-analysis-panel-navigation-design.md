# 图谱分析面板导航设计

> 点击图谱分析面板中的入口概念和路径，使左侧图谱聚焦到对应概念/路径，右侧展示概念详情。

---

## 核心理念

分析面板不仅是信息展示，更是图谱的导航入口。入口概念点击后聚焦到单个概念及其关联子图；路径点击后精确显示路径上的所有概念，形成"图谱即导航"的交互闭环。

---

## 1. 交互模型

### 点击入口 / 枢纽节点

```
用户点击 "入口" 中的 "Attention"
  ↓
onNavigate("attention_001")
  ↓
KnowledgeGraph.selectedConcept = attention_001
  focusEnabled=true → 隐藏无关节点
  → 只显示 Attention + 其直接邻居（排除 related）
  → fit 到子图
  ↓
右侧面板切换到 ConceptDetailPanel(concept=attention_001)
```

### 点击路径（最长路径/依赖链）

```
用户点击 "最长路径" 中的 "A → B → C → D → E（5步）"
  ↓
onPathFocus(["a_id", "b_id", "c_id", "d_id", "e_id"])
  ↓
KnowledgeGraph.focusedNodeIds = ["a_id", "b_id", "c_id", "d_id", "e_id"]
  → 精确显示这 5 个节点及它们之间的边
  → 第一个节点 (a_id) 作为选中节点（高亮）
  → 其他 4 个正常显示
  → fit 到刚好包围所有聚焦节点
  ↓
右侧面板切换到 ConceptDetailPanel(concept=A)
```

---

## 2. AnalysisPanel 改动

### 新增 Props

```typescript
interface AnalysisPanelProps {
  onNavigate?: (conceptId: string) => void
  onPathFocus?: (conceptIds: string[]) => void
}
```

### 入口项（Root Concepts）

- `<div>` → `<button>` + `cursor-pointer`
- `onClick={() => onNavigate?.(r.id)}`
- hover: 背景从 `bg-blue-50/60` 加深到 `bg-blue-100`
- `title`属性标记"点击聚焦到图谱"

### 路径项（最长路径/依赖链）

- 路径名/展开按钮区域 `onClick={() => onPathFocus?.(p.pathIds)}`
- 整行可点击（不只是展开按钮），展开/折叠由独立的小箭头按钮控制
- hover: 背景变灰 `hover:bg-gray-50`（与现有逻辑一致，改为 pathFocus 不冲突）
- `title`属性标记"点击在图谱聚焦路径"

### 依赖链（Dependency Chains）

与最长路径相同的交互模式。

---

## 3. KnowledgeGraph 改动

### 新增 `focusedNodeIds` Prop

```typescript
interface KnowledgeGraphProps {
  // ... existing props
  focusedNodeIds?: string[]   // 当有值时，精确指定显示的节点集合
}
```

### 聚焦逻辑变更

当前 effect（第 402-475 行）只有 `selectedConcept + focusEnabled` 的单概念 auto-neighbor 模式。

新增分支逻辑：

```
if (focusedNodeIds) {
  // 精确聚焦模式：只显示指定节点集合
  nodes: 在 focusedNodeIds 中的 → 显示，其他 → display:none
  edges: source 和 target 都在 focusedNodeIds 中的 → 显示，其他 → display:none
  选中节点: focusedNodeIds[0] 作为 selected（高亮 + 放大）
  fit: 包住所有 focusedNodeIds 中的节点（padding 60）
} else if (selectedConcept && focusEnabled) {
  // 原有 auto-neighbor 逻辑不变
}
```

### 状态一致性

- `focusedNodeIds` 和 `selectedConcept` 可以同时有值（路径模式）
- 当 `focusedNodeIds` 变更时，覆盖常规的 auto-neighbor 聚焦
- 当 `focusedNodeIds` 设为 `undefined`，回退到常规聚焦行为

---

## 4. KnowledgeGraphView 改动

### 分析面板渲染

```tsx
// 当前
<AnalysisPanel />

// 改为
<AnalysisPanel
  onNavigate={handleNavigate}
  onPathFocus={(ids) => {
    // 设置 focusedNodeIds 给 KnowledgeGraph
    setFocusedNodeIds(ids)
    // 导航到第一个概念（右侧展示详情）
    const first = concepts.find(c => c.id === ids[0])
    if (first) handleSelectConcept(first)
  }}
/>
```

### KnowledgeGraph 调用

```tsx
<KnowledgeGraph
  // ... existing props
  focusedNodeIds={focusedNodeIds}   // 新增 state
  focusEnabled={true}
/>
```

### 状态重置

- 当用户点击图谱空白区域或选择其他节点时，`focusedNodeIds` 重置为 `undefined`
- 进入过程画布（ProcessCanvas）时重置
- `handleSelectConcept` 被调用时重置

---

## 5. 边界情况

| 场景 | 行为 |
|------|------|
| 路径中的边在数据中不存在 | 只显示有边的节点，不报错 |
| 路径节点数=1 | 等同于单概念聚焦 |
| focusedNodeIds 和 selectedConcept 冲突 | focusedNodeIds 优先控制显示，selectedConcept 控制高亮 |
| 路径中的两个节点间有多条边 | 全部显示 |
| 当前聚焦路径时重新点击另一路径 | 无缝切换到新路径的节点集合 |

---

## 6. 文件改动清单

| 文件 | 改动 |
|------|------|
| `src/components/AnalysisPanel.tsx` | 新增 `onNavigate`、`onPathFocus` props；入口项可点击；路径项可点击聚焦 |
| `src/components/KnowledgeGraph.tsx` | 新增 `focusedNodeIds` prop；聚焦 effect 新增路径精确模式分支 |
| `src/components/KnowledgeGraphView.tsx` | 新增 `focusedNodeIds` state；传给 KnowledgeGraph；AnalysisPanel 传入回调 |
