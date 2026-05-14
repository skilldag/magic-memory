# 聚焦视图自动自适应布局设计

## 概述

进入知识图谱聚焦视图时，自动触发一次自适应布局，让可见节点重新排列以最优填满可视区域。

## 当前行为

聚焦视图（选中概念 → 显示自身 + 邻居节点，其余隐藏）仅执行 `cy.fit()` 做视口适配，节点位置保持全图布局时的位置。导致：

- 节点分布不紧凑，出现较大空白区域
- 容器宽高比变化后布局未重构
- 聚焦的节点子集密度与全图不一致

## 设计方案

### 改动范围

**唯一修改文件**：`src/components/KnowledgeGraph.tsx`

在现有 focus effect（第 609-640 行）末尾追加自适应布局逻辑。不涉及其他文件。

### 交互流程

```
用户点击概念节点
  → focus effect 触发，隐藏非邻居节点
  → cy.fit() 定位到聚焦节点集
  → 计算可见节点数（自身 + 邻居）
  → 调用已有的 calcAdaptiveLayoutParams 计算布局参数
  → 以可见节点数为基准运行 fcose 布局（400ms 动画）
  → layoutstop → cy.fit(聚焦节点集, dynamicPadding)
  → 更新 zoomLevel
```

### 核心代码

```typescript
// 在 focus effect 末尾 cy.fit() 之后追加
if (neighborNodes.length > 0) {
  const visibleCount = neighborNodes.length + 1
  const w = containerWidth || containerRef.current?.clientWidth || 1200
  const h = containerHeight || containerRef.current?.clientHeight || 800
  const params = calcAdaptiveLayoutParams(w, h, visibleCount)
  const layout = cy.layout({
    name: 'fcose',
    quality: 'proof',
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
    cy.fit(neighborNodes.union(selectedNode), params.padding)
    setZoomLevel(cy.zoom())
  })
  layout.run()
}
```

### 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 参数计算基数 | `visibleCount`（可见节点数） | 与全图节点数不同，聚焦子集布局密度应只考虑可见节点 |
| 布局质量 | `proof` | 聚焦场景节点通常 < 20，proof 质量不影响性能 |
| 动画时长 | 400ms | 与 `handleSmartLayout` 一致，过渡平滑 |
| fit 目标 | `neighborNodes.union(selectedNode)` | 非全图，避免空白区域 |
| 依赖数组 | 无需新增 | focus effect 已包含足够依赖 |

### 边界情况

| 场景 | 行为 |
|------|------|
| 选中节点无邻居 | `neighborNodes` 为空，跳过布局 |
| 单节点聚焦 | `visibleCount = 1`，`calcAdaptiveLayoutParams` 取最小参数 |
| 容器极小 | `calcAdaptiveLayoutParams` 已含 padding/边长 clamp |
| 退出聚焦 | 不触发布局（已有 `cy.fit()` 恢复全图），保持原行为 |

### 不变范围

- 不修改 `KnowledgeGraphView.tsx`
- 不修改 store（zustand）
- 不修改 `handleSmartLayout` 回调
- 不修改初始化布局的两阶段策略
- 不修改增量更新机制
- 不修改 `calcAdaptiveLayoutParams` 函数

## 不变范围

- 不修改 `KnowledgeGraphView.tsx`
- 不修改 store（zustand）
- 不修改 `handleSmartLayout` 回调
- 不修改初始化布局的两阶段策略
- 不修改增量更新机制
- 不修改 `calcAdaptiveLayoutParams` 函数

## 不变范围

- 不修改 `KnowledgeGraphView.tsx`
- 不修改 store（zustand）
- 不修改 `handleSmartLayout` 回调
- 不修改初始化布局的两阶段策略
- 不修改增量更新机制
- 不修改 `calcAdaptiveLayoutParams` 函数

