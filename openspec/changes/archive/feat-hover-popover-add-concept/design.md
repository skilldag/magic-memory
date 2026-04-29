# 设计：图谱节点悬停浮层

## 交互流程

```
用户鼠标悬停到图上一个概念节点 N
  │
  ▼
如果 viewMode === 'explore' 且 selectedConcept !== null
  │
  ├─ 在节点 N 旁弹出浮层
  │     ┌─────────────────────────┐
  │     │ 🔗 以「概念N」延伸       │
  │     │─────────────────────────│
  │     │ 🤖 AI 生成探索          │
  │     │ ✏️ 手动添加概念          │
  │     └─────────────────────────┘
  │
  ├─ 鼠标移入浮层 → 保持显示
  ├─ 鼠标移出浮层 → 延迟 300ms 隐藏
  ├─ 点击按钮 → 打开对应对话框
  └─ 离开触发区 → 300ms 后隐藏
```

## 位置规则

- 默认浮层出现在节点右侧 +8px
- 当节点靠近容器右边界（右边界 < 浮层宽度 + 16px）→ 浮层切换到节点左侧
- 垂直方向：浮层顶部对齐节点中心，当超出容器顶部/底部时自动偏移

## 组件结构

### 新增：ConceptHoverPopover（内联于 KnowledgeGraphView）

```
Props:
  concept: Concept
  x: number            // CG 容器内 rendered x
  y: number            // CG 容器内 rendered y
  containerWidth       // 用于边界判断
  containerHeight      // 用于边界判断
  onExplore: (c) => void
  onManualAdd: (c) => void
  onClose: () => void
```

### 状态管理

全量复用已有状态：

```typescript
// 已有 → 不变
hoverConcept, actionConcept, isHoverActionsActive
showExploreDialog, showManualLinkDialog, showBatchLinkDialog
manualInput, manualRelationType, batchSuggestions, batchLoading
```

### 修改范围（仅 KnowledgeGraphView.tsx）

| 改动 | 说明 |
|------|------|
| **删除** L351-398 的固定操作栏 `absolute top-3 right-16` | 旧的操作栏不再需要 |
| **新增** `ConceptHoverPopover` 组件 | 在 `hoverConcept` 位置渲染 |
| 修改 `onHoverConcept` handler | 将 `(concept, x, y)` 传递给浮层 |
| 保留所有 dialog 状态和 handler | `handleManualAdd`, `generateBatchSuggestions`, `confirmBatchAdd` 等 |

## UI 规格

```
背景:        bg-white shadow-xl border border-gray-200 rounded-xl
内边距:      px-4 py-3
最小宽度:    180px
最大宽度:    240px
按钮：
  - AI 生成探索:  bg-blue-50 text-blue-700 hover:bg-blue-100
  - 手动添加概念:  bg-gray-50 text-gray-700 hover:bg-gray-100
隐藏延迟:     300ms (与现有 `scheduleHideHoverActions` 一致)
```

## 防抖 / 延迟策略

- mouseenter on node → 立即显示浮层（取消任何 pending hide）
- mouseleave on node → 300ms 延迟隐藏（给用户时间移入浮层）
- mouseenter on popover → 取消隐藏 timer
- mouseleave on popover → 300ms 延迟隐藏

## 边界情况

| 场景 | 行为 |
|------|------|
| 图处于聚焦模式（focus mode） | 只对可见的邻居节点生效 |
| 鼠标快速移动经过多个节点 | 只显示最后一个鼠标所在节点的浮层 |
| 多个节点密集排布 | 只显示当前节点浮层 |
| 图平移/缩放 | 浮层固定在 hover 时的坐标，不跟随（mouseleave → 重新触发） |
