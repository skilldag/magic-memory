# 知识图谱自适应布局按钮设计

## 概述

为知识图谱添加一个「自适应布局」(Smart Fit) 按钮，点击后根据容器尺寸和节点数量动态计算 fcose 布局参数，让图谱完美填满可视区域。

## 需求背景

当前图谱的布局参数（`idealEdgeLength: 160`, `nodeRepulsion: 25000` 等）是硬编码的，不随容器大小和节点数量变化。当：

- 容器很小时（右侧面板拖宽后），节点间距显得过大
- 节点数量增多时，斥力不足导致重叠
- 容器宽高比变化时，布局未能自适应

需要一个按钮让用户一键获得「当前条件下的最佳布局」。

## 设计方案

### 按钮位置

在现有缩放控件区域（图谱右上角）新增一个按钮，位于 fit（适应视图）按钮下方：

```
 ┌──────────────┐
 │      +       │  放大
 │      −       │  缩小
 │   ⟷ (fit)    │  适应视图
 │   ⊞ (smart)  │  自适应布局 (新增)
 │    80%       │  zoom 百分比
 └──────────────┘
```

- **图标**：四个方向箭头汇聚到中心的 SVG
- **Tooltip**：「自适应布局」
- **样式**：与现有按钮一致（`w-8 h-8 bg-white rounded shadow`）

### 自适应参数计算

新增 `calcAdaptiveLayoutParams` 工具函数：

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
): AdaptiveParams
```

计算公式：

| 参数 | 公式 | 上下界 |
|------|------|--------|
| `idealEdgeLength` | `containerDiagonal / sqrt(nodeCount) × 1.2` | [60, 200] |
| `nodeRepulsion` | `idealEdgeLength × nodeCount × 3` | [8000, 50000] |
| `gravity` | `100 / (nodeCount + 10)` | [0.02, 0.3] |
| `padding` | `min(containerWidth, containerHeight) × 0.06` | [20, 80] |
| `numIter` | `nodeCount × 8` | [100, 800] |

其中 `containerDiagonal = sqrt(containerWidth² + containerHeight²)`。

### 交互流程

```
用户点击按钮
  → 获取 containerRef 当前尺寸 (width, height)
  → 获取 concepts.length (节点数)
  → 若 nodeCount ≤ 1，直接 cy.fit() 后返回
  → 若 linkMode 激活，按钮 disabled
  → 计算自适应参数
  → 运行 fcose 动画布局 (animate: true, animationDuration: 400)
  → layoutstop 事件 → cy.fit(nodes, dynamicPadding)
  → 更新 zoomLevel state
```

### 边界情况

| 场景 | 行为 |
|------|------|
| 图谱为空 (0 concepts) | 按钮 disabled |
| 仅 1 个节点 | 不触发重排，直接 fit 到视图中心 |
| 聚焦模式激活 | 重排后自动恢复聚焦节点的 fit |
| 连线模式 (linkMode) | 按钮 disabled |
| 容器极小 (< 200px) | 使用最小 padding + 最短边长 |

### 代码变更

仅修改 `src/components/KnowledgeGraph.tsx`：

1. **新增 props**：从 `KnowledgeGraphView` 传入 `containerWidth` / `containerHeight`
2. **新增工具函数**：`calcAdaptiveLayoutParams()`（可放在文件顶部或独立工具模块）
3. **新增回调**：`handleSmartLayout()` 实现上述交互流程
4. **新增按钮**：在缩放控件区域新增一个按钮元素
5. **传入 containerSize**：`KnowledgeGraphView` 将 `useContainerSize` 的尺寸传递给 `KnowledgeGraph`

## 不变范围

- 不修改任何 store（zustand）
- 不修改 KnowledgeGraphView 的布局逻辑
- 不修改增量更新机制
- 不修改初始化布局的两阶段策略
