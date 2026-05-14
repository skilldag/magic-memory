# 聚焦视图深度控制设计

## 概述

在知识图谱的聚焦视图中，允许用户选择显示以当前概念为中心的若干层关联概念（depth 1/2/3/All），代替当前硬编码的仅显示直接邻居（depth 1）。

## 当前行为

`KnowledgeGraph.tsx` 的聚焦 effect（L690-693）中：

```typescript
const selectedNode = cy.getElementById(selectedConcept.id)
const connectedEdges = selectedNode.connectedEdges()
const neighborNodes = connectedEdges.connectedNodes()
const relatedNodeIds = new Set([selectedConcept.id, ...neighborNodes.map(n => n.id())])
```

仅收集选中概念的**直接邻居**（depth 1），没有配置选项。用户无法查看更深层级的关联概念。

## 设计方案

### 改动范围

| 文件 | 改动 |
|------|------|
| `src/components/KnowledgeGraph.tsx` | 新增 `focusDepth` prop；新增 BFS 遍历函数；改造聚焦 effect 的节点收集逻辑 |
| `src/components/KnowledgeGraphView.tsx` | 新增 `focusDepth` state；新增深度选择器 UI；传递 prop |

### 交互流程

```
用户点击概念进入聚焦模式（现有行为）
  → 工具栏显示深度选择器: [❶] [❷] [❸] [∞]
  → 用户点击 ❷
  → focusDepth 更新为 2
  → BFS 从选中节点向外遍历 2 层
  → 计算新的 relatedNodeIds
  → 更新节点/边显隐
  → 自动调用 fcose 自适应布局 + cy.fit()
```

### 详细实现

#### 1. BFS 遍历函数

新增在 `KnowledgeGraph.tsx` 组件外部：

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

设计要点：
- 使用 `connectedEdges().connectedNodes()` 沿**所有边类型**（depends_on / leads_to / related）双向遍历
- `visited` Set 防止重复和环
- 返回值包含中心节点本身

#### 2. KnowledgeGraph props 扩展

```typescript
interface KnowledgeGraphProps {
  // ... 现有 props
  focusDepth?: number  // 新增：聚焦深度，默认 1
}
```

#### 3. 聚焦 effect 改造

现有 effect（依赖 `[selectedConcept, focusEnabled, focusedNodeIds, structuralKey, conceptMastery]`）中，原 `relatedNodeIds` 计算（L690-693）改为：

```typescript
// 替换原有 3 行邻居收集为：
let relatedNodeIds: Set<string>

if (focusDepth === Infinity || focusDepth === undefined) {
  // depth=All 或未指定时显示全部
  relatedNodeIds = new Set(cy.nodes().map(n => n.id()))
} else {
  const focusIds = getNodesAtDepth(cy, selectedConcept.id, focusDepth)
  relatedNodeIds = new Set(focusIds)
}
```

`focusDepth` 加入 effect 依赖数组。

depth=All（∞）时直接返回所有节点 ID，避免无意义的全图遍历。

#### 4. 节点/边显隐逻辑不变

原有基于 `relatedNodeIds` 的节点 `display` 控制、边 `display`/`opacity` 控制、以及 `cy.fit()` 逻辑完全复用。仅改变 `relatedNodeIds` 的来源。

#### 5. 深度选择器 UI

在 `KnowledgeGraphView.tsx` 的顶部工具栏区域新增控件，条件渲染：

```tsx
{selectedConcept && (
  <div className="flex items-center gap-1 ml-2 shrink-0">
    <span className="text-[11px] text-gray-400 mr-0.5">深度</span>
    {[1, 2, 3, Infinity].map(d => (
      <button
        key={d === Infinity ? 'all' : d}
        onClick={() => setFocusDepth(d)}
        className={`w-6 h-6 rounded text-xs font-medium transition-colors ${
          focusDepth === d
            ? 'bg-blue-500 text-white'
            : 'bg-white text-gray-600 hover:bg-gray-100'
        }`}
      >
        {d === Infinity ? '∞' : d}
      </button>
    ))}
  </div>
)}
```

位置在顶部搜索栏右侧，「聚焦: xxx」标签旁边。

#### 6. 状态管理

`KnowledgeGraphView.tsx` 新增：

```typescript
const [focusDepth, setFocusDepth] = useState<number>(1)
```

传递给 `KnowledgeGraph`：

```tsx
<KnowledgeGraph
  focusDepth={focusDepth}
  // ... 其他 props
/>
```

### 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 遍历方向 | 双向（所有边类型） | 用户需要看到聚焦概念的全部关联，不分方向 |
| UI 控件类型 | 按钮组（1/2/3/∞） | 简单直观，一目了然，无需下拉/滑块 |
| 控件位置 | 顶部工具栏，聚焦标签旁 | 与上下文相关联，用户聚焦时自然看到 |
| 默认值 | depth=1 | 与现有行为完全兼容，不改变已有体验 |
| 布局处理 | 每次 depth 变化自动 re-layout | 复用已有 fcose 自适应布局机制 |
| BFS 实现位置 | KnowledgeGraph 内部（组件外纯函数） | 依赖 Cytoscape 实例，不放外部 utils |

### 边界情况

| 场景 | 行为 |
|------|------|
| depth=1（默认） | 完全等同于现有行为，无感知变化 |
| depth=2/3 但该深度无节点 | 仅显示到实际有节点的深度（visited 集不变） |
| depth=All | 显示所有节点，相当于退出聚焦但保留选中高亮 |
| 切换深度时节点剧增 | fcose 自适应布局自动处理，`calcAdaptiveLayoutParams` 已含密度适配 |
| 图中有环 | `visited` Set 天然防环，不会死循环 |
| 未选中概念 | 深度选择器不渲染 |
| 首次加载、聚焦模式未启用 | `focusDepth` 不生效（因 effect 中 `!selectedConcept || !focusEnabled` 分支提前 return） |

### 不变范围

- 不修改非聚焦模式的节点/边显隐逻辑
- 不修改 `precision focus` 模式（`focusedNodeIds` prop）
- 不修改边缘样式、颜色、图例
- 不修改 `useContainerSize` / `relayoutKey` 机制
- 不修改初始化布局和增量更新 effect
- 不修改 store/数据层

### 后续可扩展

- 遍历方向过滤：可增加「仅向下（leads_to）/仅向上（depends_on）」过滤按钮
- 保存用户偏好的 depth 到 localStorage
