# 图谱边删除 — 设计文档

> 在知识图谱中支持通过「Hover 边 + 按 Delete 键」的方式删除两个概念之间的关联关系。

---

## 1. 现状分析

当前知识图谱的边操作能力：

| 操作 | 支持情况 |
|------|---------|
| 添加边 | ✅ 通过连线模式（linkMode）手动连接，或通过 AI 批量生成 |
| 删除边 | ❌ **不支持** — 没有任何删除边的 UI 或 API |
| 删除概念 | ✅ Hover 节点时出现「删」按钮，会连带删除关联的边 |

**问题**：当用户误创建了关联关系，或想调整图谱结构时，无法单独删除某条边。唯一的变通方案是删除概念再重新添加，而这会丢失概念本身的数据。

## 2. 设计目标

- 用户 Hover 到边上时，边高亮显示（视觉反馈）
- 按 Delete 键直接删除该边，无需确认
- 删除后自动持久化到服务器
- 最小化代码变更，不引入新依赖

## 3. 交互流程

```
鼠标移到边上
  → 边高亮（变粗、变色）
  → 用户按下 Delete 键
    → 从 store 移除该边
    → 边从图谱消失
    → 持久化到服务器
  → 鼠标移出边
    → 边恢复原样式（如果是按 Delete 前移出）
```

## 4. 视觉反馈

Hover 边时的样式变化（基于现有 Cytoscape 样式扩展）：

| 属性 | 默认值 | Hover 值 |
|------|--------|---------|
| `width` | 2 / 3（按类型） | 6 |
| `opacity` | 0.5 / 0.9 | 1.0 |
| `line-color` | 按类型 | 加深/加亮 |
| `target-arrow-color` | 按类型 | 同 line-color |

## 5. 组件变更

### 5.1 `src/store/knowledgeGraphStore.ts` — 新增 `removeEdge`

```typescript
removeEdge: (edgeId: string) => void
```

- 按 edge.id 从 edges 数组中过滤掉该边
- 调用 `persistToServer()` 持久化

已有 `removeConcept` 作为参考，实现模式一致。

### 5.2 `src/components/KnowledgeGraph.tsx` — 核心交互

**新增 prop:**
```typescript
onDeleteEdge?: (edgeId: string) => void
```

**Edge Hover 事件（Cytoscape 初始化内）:**
- `cy.on('mouseover', 'edge', handler)` → 保存 hoveredEdgeId 到 ref，应用 hover 样式
- `cy.on('mouseout', 'edge', handler)` → 清除 hoveredEdgeId，恢复样式

**Delete 键监听:**
- `containerRef` 元素监听 `keydown` 事件（需设置 tabIndex 使其可聚焦）
- 使用 React `useEffect` 注册/清理
- 当 `hoveredEdgeRef.current` 存在且按键为 `Delete` 时 → 调用 `onDeleteEdge(edgeId)`

### 5.3 `src/components/KnowledgeGraphView.tsx` — 连接回调

```typescript
onDeleteEdge={(edgeId) => {
  useKnowledgeGraphStore.getState().removeEdge(edgeId)
}}
```

## 6. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/store/knowledgeGraphStore.ts` | 修改 | 新增 `removeEdge` action |
| `src/components/KnowledgeGraph.tsx` | 修改 | 添加 edge hover + Delete 键监听 + 样式反馈 |
| `src/components/KnowledgeGraphView.tsx` | 修改 | 传递 `onDeleteEdge` 回调 |

## 7. 非目标

- ❌ 不做确认对话框
- ❌ 不支持 Backspace 键（仅 Delete 键）
- ❌ 不修改服务器端代码
- ❌ 不新增 npm 依赖
