# 概念推导层级展开流程 — 设计文档

> 本设计是对 `design.md`（问题驱动推导学习）的补充，聚焦于「概念→子概念」的层级展开流程。
> 回答的核心问题：用户如何从一个概念自然地过渡到其子概念的推导？
>
> ⚠️ 关键约束：用户首次进入一个概念时，对该概念的「过程」是未知的。
> ⚠️ 第一原则：第一次接触由好奇驱动，不是记忆驱动。
> ⚠️ 实现约束：每个概念定制动画不可行，必须用文档+交互替代。

---

## 1. 核心理念：认知循环的嵌套

```
┌─────────────────────────────────────────────┐
│  认知循环 N（当前概念）                        │
│                                              │
│  过程 → 矛盾 → 问题 → 推导 → 揭晓概念          │
│                                              │
│  揭晓后 → 自然产生新问题：「那它的内部呢？」       │
└─────────────────────────────────┬───────────┘
                                  │ 过渡
                                  ↓
┌─────────────────────────────────────────────┐
│  认知循环 N+1（子概念）                        │
│                                              │
│  新过程 → 新矛盾 → 新问题 → 新推导 → 揭晓子概念  │
└─────────────────────────────────────────────┘
```

---

## 2. 概念面板：首次接触的信息枢纽

第一次双击进入概念画板时，不展示空白画板——而是展示**概念信息面板**。

### 2.1 面板布局

```
┌─────────────────────────────────────────────┐
│  ← 返回图谱                                   │
│                                              │
│  ┌─ 概念卡片 ────────────────────────────┐   │
│  │  VllmConfig — 配置中心                  │   │
│  │  vLLM 的配置中心，包含 model/cache/     │   │
│  │  scheduler 三个子系统配置，在初始化时    │   │
│  │  决定整个推理系统的行为。                │   │
│  └────────────────────────────────────────┘   │
│                                              │
│  ┌─ 关联上下文 ───────────────────────────┐   │
│  │  前置概念：无 (vLLM 根概念)              │   │
│  │  引出概念：Device, Tensor, Scheduler     │   │
│  │                                         │   │
│  │  基础问题：                              │   │
│  │  · 要让模型跑推理，首先需要知道什么？      │   │
│  │  · 配置分为哪几个子系统？                │   │
│  │  · 配置错误会导致什么？                  │   │
│  └────────────────────────────────────────┘   │
│                                              │
│  ┌─ 要素勾勒 ────────────────────────────┐   │
│  │  · ModelConfig     — 模型结构参数       │   │
│  │  · CacheConfig     — KV 缓存策略        │   │
│  │  · SchedulerConfig — 调度策略           │   │
│  │  · ParallelConfig  — 并行配置           │   │
│  └────────────────────────────────────────┘   │
│                                              │
│  ┌─ 知识文档 ────────────────────────────┐   │
│  │  [📖 查看完整文档] → DocumentViewer     │   │
│  └────────────────────────────────────────┘   │
│                                              │
│  ┌─ 提问区 ─────────────────────────────┐   │
│  │  对这个概念有什么疑问？                  │   │
│  │  ┌──────────────────────────────┐     │   │
│  │  │ 输入你的问题...               │     │   │
│  │  └──────────────────────────────┘     │   │
│  │  [提问]                              │   │
│  │                                      │   │
│  │  已沉淀问题：                          │   │
│  │  · CacheConfig 和 ModelConfig 的关系？  │   │
│  │  · 配置变更需要重启吗？                │   │
│  └────────────────────────────────────────┘   │
│                                              │
│  [进入过程画板]                                │
└──────────────────────────────────────────────┘
```

### 2.2 五个区域

| 区域 | 功能 | 交互 |
|------|------|------|
| 概念卡片 | 一句话说清「它是什么」 | 只读 |
| 关联上下文 | 前置/引出关系 + 基础引导问题 | 点击问题/概念名可跳转 |
| 要素勾勒 | 关键组成部分列表 | 可展开查看详情 |
| 知识文档 | 完整文档内容 | 弹出 DocumentViewer |
| 提问区 | 自由提问 | 输入→保存到问题集 |

### 2.3 基础问题

基础问题是**引导性思考题**，不需要回答。问题本身帮助建立认知框架：

```
· 要让模型跑推理，首先需要知道什么？
→ 指向 VllmConfig 的核心职责

· 配置分为哪几个子系统？
→ 指向要素列表（ModelConfig, CacheConfig...）

· 配置错误会导致什么？
→ 指向边界理解
```

基础问题可关联到目标概念，点击可跳转。

---

## 3. 过程画板：从信息到推导

用户阅读后点击「进入过程画板」——走 `design.md` 的原有流程。

### 3.1 面板 + 画板双栏协作

```
┌───────── ProcessCanvas ─────────┬─── Panel ───────┐
│                                 │                   │
│  (自由拖拽/连线/提交)            │  概念卡片 (收起)    │
│                                 │                   │
│                                 │  要素勾勒 (展开)    │
│                                 │  · ModelConfig     │
│                                 │  · CacheConfig ←   │
│                                 │  · SchedulerConfig │
│                                 │                   │
│                                 │  [📖 文档]         │
│                                 │  [💬 提问]         │
│                                 │                   │
└─────────────────────────────────┴───────────────────┘
```

右侧面板在画板模式下可收起/展开，作为推导时的参考对照。

---

## 4. 提问驱动：问题→概念的生长

这是最核心的新机制。

### 4.1 提问

用户随时可以提问（概念面板、画板、对照验证）：

```
用户输入：「CacheConfig 和 ModelConfig 有什么关系？」

→ 保存到该概念的问题集
→ 记录提问时的上下文（位置、关联步骤）
```

### 4.2 问题的三种去向

```
                    ┌─ 留在问题集（供复习回顾）
  用户提问 ─────→  ├─ 转换为新概念（创建图谱节点）
                    └─ 合并为过程步骤（补充 process chain）
```

**留在问题集**（默认）：问题出现在问题列表中，可被查看和回答。

**转换为新概念**：
```
提问：「CacheConfig 和 ModelConfig 的配置项是怎么组织的？」

如果问题指向未建模的子领域：
→ 建议用户：「这个问题指向一个新的子概念，要创建吗？」
→ 用户确认 → 创建 Concept 节点
→ 新概念自动获得:
    title = 从问题提取
    problem = 用户问题原文
    depends_on = [VllmConfig]
    baseQuestions = 自动生成
→ 新概念出现在图谱中
```

**合并为过程步骤**：
```
用户在推导中发现少了「配置校验」：
→ 提问：「VllmConfig 加载后做配置校验吗？」

如果 ProcessChain 中没有该步骤：
→ 建议用户：「要添加'配置校验'为过程步骤吗？」
→ 用户确认 → 补充到 ProcessChain.steps
→ 该步骤下次出现在参考流程中
```

### 4.3 问题集数据模型

```typescript
interface ConceptQuestion {
  id: string
  conceptId: string
  question: string
  askedBy: string
  askedAt: Date
  context: {
    location: 'concept_panel' | 'process_canvas' | 'comparison'
    stepId?: string
  }
  status: 'open' | 'converted_to_concept' | 'converted_to_step' | 'resolved'
  convertedTo?: {
    type: 'concept' | 'step'
    targetId: string
  }
  answers: { content: string; author: string; createdAt: Date }[]
}
```

---

## 5. 子概念过渡

### 5.1 过渡规则

| 入口 | 子概念状态 | 跳转目标 |
|------|-----------|----------|
| 关联上下文点击 | 首次进入 | 概念面板 |
| 关联上下文点击 | 已进入过 | 概念面板 |
| ComparisonPanel gap | 首次进入 | 概念面板 |
| ComparisonPanel gap | 已进入过 | 画板 |
| 图谱双击 | — | 概念面板 |

进入子概念的概念面板时，关联上下文显示来自父概念的引用：

```
┌─ 关联上下文 ─────────────────────────┐
│                                      │
│  来自「VllmConfig」：                 │
│  「配置好了，模型要在什么硬件上执行？」   │
│                                      │
│  前置概念：VllmConfig                 │
│  引出概念：CUDA, VRAM                 │
│                                      │
└──────────────────────────────────────┘
```

### 5.2 用户完整旅程

```
Step 1: 双击 vLLM 节点 → 进入概念面板
Step 2: 阅读概念卡片 → 浏览要素勾勒 → 查看关联上下文
Step 3: 对「Device」感兴趣 → 点击跳转到 Device 面板
Step 4: Device 面板显示「来自 vLLM: 配置好了，模型要在什么硬件上执行？」
Step 5: 回到 vLLM → 进入过程画板
Step 6: 拖拽节点、连线 → 提交 → ComparisonPanel 显示 gap
Step 7: gap 点击 → 进入子概念面板 → 继续
Step 8: 在面板中想到一个问题 → 输入 → 沉淀到问题集
Step 9: 问题 → 转化为新概念 → 出现在图谱中
```

---

## 6. 回溯与结构感知

### 6.1 面包屑

```
图谱  >  VllmConfig [面板]  >  Device [面板]
```

### 6.2 层级进度树

```
vLLM ───────────────── 已查看 3/5 ──
├─ VllmConfig  ✅ 已推导
├─ Device      🔄 已查看 (当前)
│  ├─ CUDA     ⬜ 未查看
│  └─ VRAM     ⬜ 未查看
├─ Tensor      🔄 已查看
└─ Scheduler   ⬜ 未查看
```

### 6.3 自动保存

面板展开状态、画板状态、提问记录分别保存。

---

## 7. 数据模型

```typescript
// 面板展开状态
interface ConceptPanelState {
  conceptId: string
  sectionsExpanded: {
    card: boolean; context: boolean
    elements: boolean; doc: boolean; questions: boolean
  }
}

// 基础问题
interface BaseQuestion {
  id: string
  conceptId: string
  question: string
  targetConceptId?: string
  hint?: string
}

// Concept 扩展
interface Concept {
  hierarchy: { parentId: string | null; level: number; order: number }
  baseQuestions?: BaseQuestion[]
}

// Store
interface CanvasStore {
  conceptPanelStates: Map<string, ConceptPanelState>
  questions: ConceptQuestion[]
  canvasHistory: { conceptId: string; view: 'panel' | 'canvas' }[]
  hierarchyProgress: HierarchyProgress
}
```

---

## 8. 与 design.md 的关系

| 本设计内容 | 对应 design.md | 关系 |
|-----------|---------------|------|
| 概念面板 | 新增首次进入状态 | 替代空白画板作为默认入口 |
| 关联上下文 + 基础问题 | — | 新增 |
| 要素勾勒 | §6 ConceptElement | 复用已有数据 |
| 知识文档对照 | §2.2 DocumentViewer | 复用 |
| 提问机制 | — | 新增 |
| 过程画板 | §3 ProcessCanvas | 保留不变 |
| ComparisonPanel | §4 | 保留不变 |

---

## 9. 不做的事

- 不做过程动画/自动播放
- 不改变 ProcessCanvas 和 ComparisonPanel 的交互
- 问题→新概念/步骤的转化由用户确认
- 不修改知识图谱的 Cytoscape 渲染
