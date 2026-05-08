# 知识图谱全局搜索 — 设计文档

> 在知识图谱中添加全局搜索功能，允许用户通过概念标题、别名和问题描述快速定位到目标概念。

---

## 1. 现状分析

当前知识图谱的交互方式：
- **单击节点** → 右侧面板展示概念详情
- **双击节点** → 进入过程画板
- **图谱摘要面板**（无选中时）→ 显示入口节点、枢纽节点、最长路径

**问题**：当图中概念数量较多（50+）时，用户难以快速定位到特定概念。没有搜索功能，只能靠视觉扫描或通过关联关系一步步导航。

## 2. 设计目标

- 允许用户通过输入关键词快速搜索图谱中的所有概念
- 客户端实时过滤，零延迟响应
- 点击搜索结果直接聚焦到该概念（选中 + 图跳跃）
- 最小化代码变更，不引入新依赖

## 3. 搜索数据流

```
用户输入 query
  → knowledgeGraphStore.concepts 实时过滤
    → 匹配: title / alias / problem 字段（大小写不敏感）
    → 排序: 精确匹配 > 前缀匹配 > 包含匹配 > 别名匹配 > 问题匹配
  → 结果列表（上限 10 条）
  → 用户点击/回车确认
    → selectConcept(concept) + 图谱聚焦到该节点
```

搜索完全在客户端进行，不依赖后端。所有概念数据已在 Zustand store 中。

## 4. UI 布局

搜索输入框放置在图谱视图的顶部焦点栏区域（选中概念时）或图谱区域上方（无选中时）。

```
┌───────────────────────────────────────────────┐
│ 🔍 [搜索概念...                  ]  (搜索输入框) │
│  ┌───────────────────────────────────────────┐ │
│  │ 📘 PagedAttention        · L2 · Attention │ │
│  │ 📘 PageAttention         · L2 · Attention │ │
│  │ 📘 Block Table           · L2 · Attention │ │
│  │ 📘 KVCacheManager        · L3 · Schedule  │ │
│  └───────────────────────────────────────────┘ │
├───────────────────────────────────────────────┤
│                                                │
│           知识图谱 (Cytoscape.js)               │
│                                                │
└───────────────────────────────────────────────┘
```

### 4.1 搜索输入框

- 左侧放大镜 SVG 图标
- Placeholder: "搜索概念..."
- 输入时即时过滤，无防抖（客户端操作，零延迟）

### 4.2 下拉结果面板

- 浮在图谱上方（`z-index` 高于图谱）
- 白色背景 + 阴影 (`shadow-lg`)
- 最多显示 10 条结果
- 每条结果显示: 概念标题、Level 标签、分类标签
- 无结果时显示 "未找到匹配的概念"
- 空输入时隐藏

### 4.3 键盘交互

| 按键 | 行为 |
|------|------|
| ↑ / ↓ | 移动高亮行 |
| Enter | 选中高亮的概念 |
| Esc | 关闭下拉，清空输入 |
| 点击外部 | 关闭下拉 |

## 5. 匹配算法

```typescript
function matchScore(concept: Concept, query: string): number {
  const q = query.toLowerCase()
  const title = concept.title.toLowerCase()
  const aliases = concept.alias?.map(a => a.toLowerCase()) ?? []
  const problem = concept.problem?.toLowerCase() ?? ''

  if (title === q) return 100      // 精确匹配
  if (title.startsWith(q)) return 80  // 前缀匹配
  if (title.includes(q)) return 60    // 包含匹配
  if (aliases.some(a => a.includes(q))) return 40  // 别名匹配
  if (problem.includes(q)) return 20  // 问题描述匹配
  return 0  // 不匹配
}
```

按 score 降序排列，取前 10 条。score 为 0 的过滤掉。

## 6. 组件架构

### 6.1 新建 `src/components/GlobalSearch.tsx`

搜索组件，零外部依赖，纯展示逻辑。

**Props**:
- `concepts: Concept[]` — 要搜索的概念列表
- `onSelect: (concept: Concept) => void` — 选中回调

**内部状态**:
- `query: string` — 输入文本
- `isOpen: boolean` — 下拉是否可见
- `highlightIndex: number` — 键盘导航高亮索引

**功能**:
- 输入框 onChange → 过滤 concepts → 更新结果列表
- ↑↓ 键导航 highlightIndex
- Enter 键选中当前高亮项
- Esc 键关闭 + 清空
- 点击外部区域 → 关闭下拉

### 6.2 修改 `src/components/KnowledgeGraphView.tsx`

在顶部焦点栏区域插入 `<GlobalSearch />`。

变化点:
- 在焦点栏 `selectedConcept && !processMode` 的 `<div>` 内加入搜索组件
- `onSelect` → 调用 `handleSelectConcept(concept)`

## 7. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/components/GlobalSearch.tsx` | **新建** | 搜索组件 |
| `src/components/KnowledgeGraphView.tsx` | 修改 | 集成搜索组件到顶部栏 |

## 8. 非目标（明确不做）

- ❌ 不搜索文档正文 content（后续可扩展）
- ❌ 不做服务器端搜索 API
- ❌ 不做 AI 语义搜索
- ❌ 不修改后端代码
- ❌ 不修改 Zustand store
- ❌ 不新增 npm 依赖
