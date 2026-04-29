# 设计：骨架填充流程

> 首次进入概念画板时，展示预置骨架（空缺+问题），用户拖拽填充，提交验证，提问生长。

---

## 1. 骨架画板

### 初始状态

```
┌────────────────────────────────────────────────────┐
│  vLLM  ·  过程梳理画板                                │
│                                                      │
│  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐     │
│  │ ❓   │──→│ ❓   │──→│ ❓   │──→│ ❓   │     │
│  │ 配置?│    │ 设备?│    │ 数据?│    │ 运行?│     │
│  └──────┘    └──────┘    └──────┘    └──────┘     │
│     ↑           ↑           ↑           ↑          │
│  配置了    初始化在   数据怎么   谁负责               │
│  什么？    什么硬件？  表示？     执行？              │
│                                                      │
│  ┌─ 候选概念区 ───────────────────────────┐         │
│  │  [VllmConfig] [Device] [Tensor] ...     │         │
│  │  [+ 自定义]                 [💬 提问]   │         │
│  └─────────────────────────────────────────┘         │
│                                                      │
│  [提交验证]   4空缺 · 0已填                          │
└──────────────────────────────────────────────────────┘
```

空缺节点展示引导问题（来自 `BaseQuestion`），用户从候选区拖拽概念填入。

### 两种入口

| 入口 | 子概念状态 | 跳转目标 |
|------|-----------|----------|
| 图谱双击 | 首次进入 | 骨架画板 |
| ComparisonPanel gap | 首次进入 | 骨架画板（带父上下文） |
| 任意入口 | 已进入过 | 自由画板 |

### 提交验证

提交后逐项比对 `filledConceptId === correctConceptId`：

```
✓ VllmConfig   正确
⚡ Tensor      错误 → 正确是 GpuAllocator
⚡ 空缺 4      未填充 → [去查看]
填写率: 75%
```

提交后进入 `design.md` 原有的 ComparisonPanel 流程。

---

## 2. 提问机制

用户随时可通过 💬 按钮提问，问题有三种去向：

```
                    ┌─ 留在问题集（复习回顾）
  用户提问 ─────→  ├─ 转化为新概念（创建图谱节点）
                    └─ 补充为流程步骤（扩展 skeleton）
```

### 数据模型

```typescript
// 预置引导题 — 驱动骨架填充
interface BaseQuestion {
  id: string
  conceptId: string
  question: string          // "要让模型跑推理，首先需要知道什么？"
  targetConceptId?: string  // 指向正确答案
  hint?: string
  order: number
}

// 用户提问 — 可转化为新概念或步骤
interface UserQuestion {
  id: string
  conceptId: string
  question: string
  context: { location: 'skeleton' | 'canvas' | 'comparison'; stepId?: string }
  status: 'open' | 'converted_to_concept' | 'converted_to_step' | 'resolved'
  convertedTo?: { type: 'concept' | 'step'; targetId: string }
  createdAt: Date
}
```

---

## 3. 回溯与结构感知

### 面包屑

```
图谱  >  VllmConfig [填充]  >  Device [填充]
```

### 层级进度树

```
vLLM ──────────────── 已掌握 5/8 ──
├─ VllmConfig  ✅ 已掌握
├─ Device      🔄 填充中 (当前)
│  ├─ CUDA     ⬜ 未开始
│  └─ VRAM     ⬜ 未开始
└─ Tensor      ✅ 已掌握
```

---

## 4. 与现有代码的关系

| 组件 | 变更 |
|------|------|
| `types/index.ts` | 新增 `BaseQuestion`, `UserQuestion`, `CanvasHistoryItem`；扩展 `Concept.hierarchy` |
| `data/mockGraphData.ts` | 新增 `mockBaseQuestions`，补充到 mock concepts |
| `utils/processComparison.ts` | 新增 `generateSkeletonNodes()` |
| `store/knowledgeGraphStore.ts` | 新增 `questions`, `canvasHistory`, `skeletonCompleted` |
| `components/ProcessCanvas.tsx` | 新增 `skeletonMode` prop + 骨架模式渲染 |
| `components/KnowledgeGraphView.tsx` | 新增面包屑栏、骨架模式控制、提问弹窗 |
| `components/ConceptDetailPanel.tsx` | 新增「问题集」tab |
