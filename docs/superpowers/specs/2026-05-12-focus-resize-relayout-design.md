# 聚焦视图拉伸后自适应布局设计

## 概述

拖拽右侧面板分割线拉伸聚焦视图后，在鼠标抬起时自动触发一次图谱自适应布局，让节点在新容器尺寸下重新排列。

## 当前行为

聚焦视图右侧面板（ConceptDetailPanel）可通过拖拽分割线左右拉伸。图谱区域的 `containerWidth`/`containerHeight` 虽随 ResizeObserver 更新，但图谱节点位置不变，仅靠 `cy.fit()` 做视口适配。导致：

- 拉伸后布局密度不适合新尺寸（节点间距过密或过疏）
- 容器宽高比变化后布局未重构
- 需要用户手动点击「自适应布局」按钮

## 设计方案

### 改动范围

| 文件 | 改动 |
|------|------|
| `src/components/KnowledgeGraph.tsx` | 新增 `relayoutKey` prop + watch effect |
| `src/components/KnowledgeGraphView.tsx` | 新增 `relayoutKey` state + mouseup 时递增 |

### 交互流程

```
用户拖拽分割线调整右侧面板宽度
  → 鼠标抬起 (mouseup)
  → mouseup handler 检测聚焦模式 (selectedConcept != null)
  → setRelayoutKey(k => k + 1)
  → KnowledgeGraph watch effect 识别 relayoutKey 递增
  → 调用 handleSmartLayout (fcose, proof quality, 400ms 动画)
  → layoutstop → cy.fit(undefined, dynamicPadding)
  → 更新 zoomLevel
```

### 详细实现

#### KnowledgeGraph.tsx

新增 prop 定义：

```typescript
interface KnowledgeGraphProps {
  // ... 现有 props
  relayoutKey?: number  // 递增时触发自适应布局
}
```

新增 watch effect：

```typescript
const handleSmartLayoutStable = useRef(handleSmartLayout)
handleSmartLayoutStable.current = handleSmartLayout

useEffect(() => {
  if (relayoutKey && relayoutKey > 0) {
    handleSmartLayoutStable.current()
  }
}, [relayoutKey])
```

使用 ref 兜底避免闭包陈旧问题（effect 中引用函数可能不是最新）。

#### KnowledgeGraphView.tsx

新增 state：

```typescript
const [relayoutKey, setRelayoutKey] = useState(0)
```

传递给 KnowledgeGraph：

```tsx
<KnowledgeGraph
  // ... 现有 props
  relayoutKey={selectedConcept ? relayoutKey : 0}
/>
```

`selectedConcept` 为 null 时传 0，避免非聚焦模式下浪费布局计算。

在 `handleDividerMouseDown` 的 `handleMouseUp` 末尾追加：

```typescript
const handleMouseUp = () => {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
  // 聚焦模式下触发布局
  if (selectedConceptRef.current) {
    setRelayoutKey(k => k + 1)
  }
}
```

### 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 触发方式 | `relayoutKey` prop 递增触发 | React 原生模式，无 ref/forwardRef 侵入 |
| 触发时机 | 仅 mouseup | 拖拽结束一次，不在拖拽过程中反复计算 |
| 触发条件 | 仅 `selectedConcept != null` | 按需求仅在聚焦模式 |
| 布局函数 | 复用 `handleSmartLayout` | fcose 根据当前 containerWidth/Height 重新计算 |
| 动画 | 400ms animate | 与现有 smart layout 保持一致平滑过渡 |
| 闭包安全 | useRef 存储函数引用 | 避免 effect 依赖列表中函数的引用稳定性问题 |

### 边界情况

| 场景 | 行为 |
|------|------|
| 非聚焦模式拖拽 | `relayoutKey` 传 0，watch effect 不触发 |
| 聚焦模式但节点极少（≤1） | `handleSmartLayout` 内部已处理：`cy.fit()` |
| 拖拽到最小/最大宽度 | 正常触发，`calcAdaptiveLayoutParams` 已含 clamp |
| 多次快速拖拽 | 每次 mouseup 触发一次，fcose 会打断上一次动画 |
| 初始加载时 | `relayoutKey` 为 0，不触发 |

### 不变范围

- 不修改 `handleSmartLayout` 内部逻辑
- 不修改 `calcAdaptiveLayoutParams`
- 不修改 `useContainerSize` / `usePanelResizing`
- 不修改初始化布局的两阶段策略
- 不修改增量更新机制
- 不修改聚焦/非聚焦的节点显隐逻辑
