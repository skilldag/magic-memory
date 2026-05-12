# 记忆曲线复习提醒 — 设计文档

> 在知识图谱上直观标记概念的复习状态，通过 Badge + 语义对齐流程实现间隔重复提醒，让用户不遗漏任何到期复习的概念。

---

## 1. 现状分析

### 1.1 已有基础设施

| 能力 | 状态 | 位置 |
|------|------|------|
| SM-2 间隔重复算法 | ✅ 已实现 | `store/knowledgeGraphStore.ts` → `startReview()` |
| `ReviewRecord` 类型 | ✅ 已定义 | `types/index.ts`（含 `interval`, `ease_factor`, `next_review`, `status`） |
| 节点颜色按掌握分映射 | ✅ 已实现 | `KnowledgeGraph.tsx` 使用 `conceptMastery` 着色 |
| 语义对齐面板 | ✅ 已实现 | `AlignmentPanel.tsx`，可计算用户理解匹配度 |
| 图谱摘要面板 | ✅ 已实现 | `SummaryPanel.tsx`，展示图谱概览 |

### 1.2 缺失能力

- **节点无复习状态标记**：用户无法从图谱上直观看出哪些概念需要复习
- **无复习入口**：打开应用时不知道有待复习概念
- **复习与对齐未联动**：对齐完成后不会自动触发 SM-2 复习记录
- **无遗忘曲线可视化**：看不到每个概念的复习间隔增长趋势

---

## 2. 设计目标

- **节点 Badge**：每个概念节点右上角标记复习状态（`🔥` 过期 / `今日` 到期 / `Xd` 即将到期 / `NEW` 未学）
- **复习横幅**：进入图谱时，如有过期概念，显示待复习数量横幅
- **点击 → 对齐复习**：点击带复习标记的节点，右侧面板自动切到「语义对齐」，对齐完成后自动触发 SM-2
- **复习待办区**：SummaryPanel 中新增「📅 复习待办」区块，按紧急度排序
- **遗忘曲线迷你图**：展示每个概念的复习间隔增长趋势

---

## 3. Badge 体系

### 3.1 状态定义

基于 `ReviewRecord` 的 `next_review` 计算：

| 条件 | Badge | 语义颜色 | 示例 |
|------|-------|---------|------|
| 无 `ReviewRecord` | `NEW` | `#6b7280` 灰 | 新概念 |
| `next_review < now` | `🔥` | `#ef4444` 红 | 已过期 |
| `next_review` 在今天内 | `今日` | `#f59e0b` 琥珀 | 今天到期 |
| `next_review` 在未来 1-7 天 | `Xd` | `#3b82f6` 蓝 | 即将到期 |
| `interval > 21` 且 `status=mastered` | `✓` | `#10b981` 绿 | 已掌握 |
| 刚刚复习完（`last_reviewed < 1h`） | 不显示 | — | 减少干扰 |

### 3.2 渲染方式

利用 Cytoscape 已有节点样式扩展：

```typescript
// 在 KnowledgeGraph.tsx 中，对每个节点追加 badge 样式
// 通过 Cytoscape 的 node label 或叠加 HTML 层实现

// 方式：Cytoscape 的 node 内嵌 div 叠加层
// 在 cy.node().data() 中增加 badge 字段
// 通过 node.style 的 'border-style' 或叠加 'overlay' 来实现
```

使用 Cytoscape 的 `overlay` 或 `border` 样式 + 叠加 badge 文本。为避免和已有的 mastery 颜色冲突，Badge 画在节点右上角的叠加层中，mastery 颜色作为节点背景色不变。

### 3.3 工具函数

```typescript
// src/utils/knowledgeGraph.ts

type ReviewBadge = {
  text: string       // 显示文本: "🔥" | "今日" | "1d" | "NEW" | "✓" | ""
  color: string      // Badge 背景色
  urgency: number    // 排序用：0=过期 1=今日 2=3天内 3=7天内 4=已掌握 5=无记录
}

function getReviewBadge(record?: ReviewRecord): ReviewBadge

function getDueConcepts(
  concepts: Concept[], records: Map<string, ReviewRecord>
): { concept: Concept; badge: ReviewBadge; daysUntilReview: number }[]
// 返回所有待复习概念，按紧急度排序
```

---

## 4. 复习入口横幅

### 4.1 位置

`KnowledgeGraphView.tsx` 中，搜索栏下方。

### 4.2 触发条件

- 进入 `KnowledgeGraphView` 时
- 有任意概念 `next_review < now`
- 会话中已确认过不再显示后，可手动关闭

### 4.3 UI

```
┌─────────────────────────────────────────────────────────┐
│ 📅 5 个概念需要复习 · 最长的已过期 3 天  [查看待复习 →]  [×] │
└─────────────────────────────────────────────────────────┘
```

- 查看待复习 → 选中过期最久的概念，右侧面板展示，自动切到「语义对齐」
- `[×]` 关闭横幅（session 内不再显示）

### 4.4 状态管理

```
bannerDismissed: boolean  // sessionStorage 或 useState
// 仅在 KnowledgeGraphView 内维护，不清除 store
```

---

## 5. 点击 → 语义对齐流程

### 5.1 交互决策树

```
用户点击知识图谱节点
        │
        ▼
  检查该概念 ReviewBadge
        │
   ┌────┼────┐
   ▼    ▼    ▼
  🔥/今日  NEW  其他/✓/无
   │    │    │
   ▼    ▼    ▼
 自动切到  自动切到  保持现有行为
 「语义对齐」 「查阅文档」 （查阅文档）
   │
   ▼
 对齐完成 → 计算匹配率
   │
   ▼
 匹配率 > 80% → quality: 4
 匹配率 50-80% → quality: 3
 匹配率 < 50% → quality: 2
   │
   ▼
 调用 startReview(conceptId, quality)
   │
   ▼
 Badge 自动刷新（重新计算 getReviewBadge）
```

### 5.2 对齐触发 SM-2

```typescript
// AlignmentPanel.tsx 中，对齐完成后追加
const store = useKnowledgeGraphStore.getState()
const conceptId = concept.id
const result = alignmentResult  // GraphAlignmentResult

// 从对齐结果计算 quality
const coverage = result.stats.nodeCoverage  // 0-1
let quality: number
if (coverage > 0.8) quality = 4
else if (coverage > 0.5) quality = 3
else quality = 2

store.startReview(conceptId, quality)
// 更新 mastery (复用已有逻辑)
const newScore = Math.round(coverage * 100)
store.updateMastery(conceptId, newScore)
```

### 5.3 ConceptDetailPanel 改动

```typescript
// 新增: 根据 reviewStatus 决定默认 tab
const reviewBadge = getReviewBadge(reviewRecords.get(concept.id))
const defaultAction: ActionKey = 
  reviewBadge.text === '🔥' || reviewBadge.text === '今日' 
    ? 'align' 
    : reviewBadge.text === 'NEW' 
      ? 'read' 
      : action // 保持当前
```

---

## 6. SummaryPanel 复习待办区

### 6.1 位置

图谱摘要右侧面板中，新增区块「📅 复习待办」，排在现有「入口」「枢纽节点」「最长路径」之后。

### 6.2 UI

```
📅 复习待办                   5 个待复习
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 PagedAttention            过期 3 天
🔥 Block Table               过期 1 天
今日 KVCacheManager          今天到期
1d  ModelLoader              明天到期
3d  Sampler                  3 天后
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在轨率 ████████░░  80% (8/10 按时复习)
```

### 6.3 交互

- 点击任一条目 → 选中该概念，聚焦图谱，右侧面板切换
- 在轨率 = `按时复习次数 / 总复习次数`（按时 = 在 `next_review` 当天或之前复习）

---

## 7. 遗忘曲线迷你图

### 7.1 位置

SummaryPanel「复习待办」区块底部。

### 7.2 实现方式

```typescript
// 纯 SVG 渲染，无额外依赖
// 输入: 每个概念的 ReviewRecord[]（复习历史）

interface ReviewHistoryPoint {
  date: Date
  interval: number  // 当次复习后的间隔（天）
}

// 绘制: SVG 折线图
// - 横轴: 复习日期
// - 纵轴: 间隔天数
// - 每条线 = 一个概念的间隔增长趋势
// - 线应呈阶梯上升（间隔递增），若出现平台或下降表示遗忘
```

### 7.3 数据来源

当前 `ReviewRecord` 只存最新一条，要画遗忘曲线需要 **复习历史**。V1 中可先不画曲线，只展示「在轨率」进度条。曲线需要增加 `reviewHistory: { date: string; interval: number }[]` 字段到 `ReviewRecord`。

**V1 不做遗忘曲线图**，改为在轨率进度条。复习历史追踪留到 V2。

---

## 8. 数据模型变更

### 8.1 ReviewRecord 扩展（V1 不需变更）

```typescript
// V2 计划: ReviewRecord 追加
reviewHistory?: {
  date: string    // ISO date
  interval: number // 复习后的间隔（天）
}[]
```

### 8.2 Store 新增工具方法

```typescript
// knowledgeGraphStore.ts

// 计算所有待复习概念（按紧急度排序）
getDueConceptIds: () => string[]
// 逻辑: 遍历 reviewRecords，筛选 next_review < now 或 next_review < now + 7d
// 按 next_review 升序排列
```

---

## 9. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/utils/knowledgeGraph.ts` | 新增函数 | `getReviewBadge()`, `getDueConcepts()` |
| `src/components/KnowledgeGraph.tsx` | 修改 | Cytoscape 节点叠加 Badge 渲染 |
| `src/components/KnowledgeGraphView.tsx` | 修改 | 添加复习横幅 + 传入 reviewRecords |
| `src/components/ConceptDetailPanel.tsx` | 修改 | 根据 Badge 状态决定默认 tab |
| `src/components/AlignmentPanel.tsx` | 修改 | 对齐完成后触发 SM-2 复习记录 |
| `src/components/SummaryPanel.tsx` | 修改 | 新增「📅 复习待办」区块 |
| `src/store/knowledgeGraphStore.ts` | 新增 | `getDueConceptIds` 计算 |

---

## 10. 不做的事

- V1 不做遗忘曲线 SVG 图（改为在轨率进度条）
- V1 不记录复习历史（reviewHistory 延期到 V2）
- 不破坏现有的节点 mastery 颜色渲染
- 不修改语义对齐的核心算法
- 不添加浏览器 Notification API 推送
- 不添加邮件/第三方通知集成

---

## 11. 迭代方向

- **V2: 遗忘曲线图**：记录每次复习的时间点和间隔，SVG 渲染间隔增长趋势
- **V2: 每日复习摘要**：进入应用时弹出复习计划卡片
- **V3: 浏览器通知**：通过 Notification API 在后台推送复习提醒
- **V3: 复习统计面板**：复习次数、在轨率趋势、薄弱概念聚合
