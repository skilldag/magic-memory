# 节点颜色关联概念掌握程度 — 设计文档

> 将知识图谱节点的颜色从 Level 分类色改为反映用户掌握程度的热力图色，让用户一眼看出哪些概念掌握得好、哪些需要巩固。

---

## 1. 现状分析

当前知识图谱节点颜色按 Level 分类：

| Level | 颜色 | 含义 |
|-------|------|------|
| 1 | `#10b981` 绿 | 基础概念 |
| 2 | `#3b82f6` 蓝 | 核心组件 |
| 3 | `#8b5cf6` 紫 | 高级特性 |

**问题**：
- 颜色只反映概念难度，不反映用户对概念的掌握程度
- 系统已有**语义对齐**能力（AlignmentPanel），能计算用户理解与原文的匹配度，但该分数仅展示在面板中，未反馈到图谱上
- 用户无法在概览层面识别自己的薄弱环节

已有能力可复用：
- `utils/alignment.ts` → `compareTexts()` 产出 `GraphAlignmentResult.stats`（含 `nodeCoverage`、`nodePrecision`）
- `components/AlignmentPanel.tsx` → 用户已可对每个概念执行图对齐
- `store/knowledgeGraphStore.ts` → Zustand store 可用于存储 mastery 数据

---

## 2. 设计目标

- **每次语义对齐产生一个掌握分**，存储到 store 中
- **节点颜色根据掌握分重新映射**，替换现有的 Level 分类色
- 保留选中态（金色边框/高亮）不变
- 首次使用者看到灰色节点（未对齐），激励用户去执行对齐
- 数据通过 Graph Server 持久化，刷新后恢复

---

## 3. 数据模型

### 3.1 新增类型

```typescript
// src/types/index.ts
export interface MasteryRecord {
  conceptId: string
  score: number          // 0-100 的综合掌握分
  lastAligned: Date      // 最近一次对齐时间
  alignmentCount: number // 对齐执行次数
}
```

### 3.2 Store 变更

```typescript
// src/store/knowledgeGraphStore.ts — 新增
conceptMastery: Map<string, MasteryRecord>

updateMastery(conceptId: string, score: number): void
// 写入 conceptMastery[conceptId]，更新 lastAligned 和 alignmentCount
```

---

## 4. 算法与评分

### 4.1 掌握分公式

```
masterScore = round(nodeCoverage × 60 + nodePrecision × 40)
```

| 指标 | 来源 | 权重 | 含义 |
|------|------|------|------|
| nodeCoverage | `matchedNodeCount / originalNodeCount × 100` | 60% | 你提到了多少原文概念（广度） |
| nodePrecision | `matchedNodeCount / userNodeCount × 100` | 40% | 你提的概念中有多少相关的（精度） |

为什么加权：覆盖广度比精度更重要——遗漏知识缺口比写多无关内容问题更大。

### 4.2 更新策略

- **首次对齐**：写入 score
- **多次对齐**：取**最近一次**分数（而非平均），鼓励持续改进
- **新增概念**：默认为未对齐（无 MasteryRecord），颜色显示灰色

---

## 5. 颜色映射

| 状态 | 条件 | 色值 | 色名 |
|------|------|------|------|
| 未对齐 | 无 MasteryRecord | `#d1d5db` | 浅灰 |
| 薄弱 | score < 40 | `#ef4444` | 红 |
| 部分掌握 | 40 ≤ score < 70 | `#f59e0b` | 琥珀 |
| 良好 | 70 ≤ score < 90 | `#10b981` | 绿 |
| 精通 | score ≥ 90 | `#059669` | 深绿 |

**替换** `src/constants/graph.ts` 中的 `LEVEL_COLORS` 为：

```typescript
export const MASTERY_COLORS: Record<string, string> = {
  unaligned: '#d1d5db',
  weak:      '#ef4444',
  partial:   '#f59e0b',
  good:      '#10b981',
  mastered:  '#059669',
}

export function getMasteryColor(score: number | undefined): string {
  if (score === undefined) return MASTERY_COLORS.unaligned
  if (score >= 90) return MASTERY_COLORS.mastered
  if (score >= 70) return MASTERY_COLORS.good
  if (score >= 40) return MASTERY_COLORS.partial
  return MASTERY_COLORS.weak
}
```

---

## 6. 数据流

```
用户在 AlignmentPanel 提交理解文字
         │
         ▼
  compareTexts() → GraphAlignmentResult
         │
         ├── stats.nodeCoverage
         └── stats.nodePrecision
         │
         ▼
  masterScore = round(nodeCoverage × 0.6 + nodePrecision × 0.4)
         │
         ▼
  store.updateMastery(conceptId, masterScore)
    → conceptMastery Map 更新
    → persistToServer() 持久化
         │
         ▼
  KnowledgeGraph 渲染
    → 读取 conceptMastery[conceptId]?.score
    → getMasteryColor(score) 决定节点背景色
```

---

## 7. UI 变更

### 7.1 图谱图例

右侧图例中的「级别颜色」区块替换为「掌握程度」区块：

```
● 灰色   → 未对齐
● 红色   → 薄弱 (<40%)
● 琥珀色 → 部分掌握 (40-70%)
● 绿色   → 良好 (70-90%)
● 深绿色 → 精通 (>90%)
```

### 7.2 AlignmentPanel 联动

对齐完成后，在「执行图对齐」按钮下方增加一行提示：

```
本次掌握分: 72/100 → 节点颜色已更新
```

---

## 8. 影响文件清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `src/types/index.ts` | 新增 | 添加 `MasteryRecord` 接口 |
| `src/store/knowledgeGraphStore.ts` | 修改 | 新增 `conceptMastery` state + `updateMastery()` action；`persistToServer()` 扩展持久化 mastery 数据 |
| `src/constants/graph.ts` | 修改 | 替换 `LEVEL_COLORS` 为 `MASTERY_COLORS` + `getMasteryColor()` 函数 |
| `src/components/KnowledgeGraph.tsx` | 修改 | 节点渲染时通过 `mastery` data 字段取色，替换 `LEVEL_COLORS` 引用 |
| `src/components/KnowledgeGraphView.tsx` | 修改 | 将 `conceptMastery` 传递给 `KnowledgeGraph` |
| `src/components/AlignmentPanel.tsx` | 修改 | 对齐完成后调用 `updateMastery()`，显示分数反馈 |

---

## 9. 不做的事

- 不改动边颜色（边按关系类型着色保持不变）
- 不改动选中态（金色边框/高亮保持不变）
- 不改动节点大小（Level 尺寸编码建议保留作为辅助信息，但不在本 spec 范围内）
- 不做多次对齐分数趋势图表（未来迭代）
- 不做掌握分排行榜或比较

---

## 10. 验证标准

- 新项目：所有节点显示为灰色
- 对一个概念执行语义对齐 → 节点颜色更新为红/琥珀/绿/深绿（取决于分数）
- 刷新页面后 → 颜色恢复（通过 Graph Server 持久化）
- 选中节点时 → 金色边框/高亮不受影响
- 未对齐概念 → 始终为灰色
