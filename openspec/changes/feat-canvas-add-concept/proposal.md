# Proposal: 图谱画布双击添加概念 + 手动关系更新

## 现状

目前用户添加概念的入口有三个：
1. **AI 自动探索** — 悬停节点出现的 `AI` 按钮 → BatchLinkDialog
2. **基于问题生成** — 悬停节点出现的 `?` 按钮 → QuickExploreDialog
3. **手动添加** — 悬停节点出现的 `手动` 按钮 → ManualAddDialog

**所有入口都依赖已有节点作为 source**，无法在空白区直接创建概念。

此外，概念的关系（depends_on/leads_to/related）只在创建时通过对话框指定一次。如果后续编辑概念的 Markdown 文档（修改 frontmatter），已有的关系不会自动或手动同步更新。

## 问题

1. **无法从空白区起步** — 如果图谱为空或用户想创建一个孤立概念起步，没有任何入口
2. **文档和关系两张皮** — 用户在 ConceptDetailPanel 编辑概念文档，改动了 frontmatter 中的关系，图谱不会同步更新
3. **缺少手动触发更新机制** — 没有"把文档中的关系写回图结构"的入口

## 方案

### Feature A: 双击空白区添加概念节点

利用 `KnowledgeGraph.tsx` 已有的 `onBackgroundDoubleTap` 回调（`cy.on('tap')` + 300ms 时间窗检测），在双击图谱空白区时弹出轻量输入对话框：

```
双击空白区 → 弹窗输入概念名 → 确认 → store.addConcept() → 增量布局
```

复用 store 中现有的 `addConcept` 和 `createConceptWithEdges` 方法，`KnowledgeGraph.tsx` 的增量更新 useEffect 自动检测新节点并运行 fcose 增量布局。

### Feature B: ConceptDetailPanel 手动关系和图谱更新

在右侧面板的文档编辑区下方新增"更新关系和图谱"按钮：

```
点击 → parseFrontmatter(当前文档 content)
     → matchTitlesToIds(depends_on/leads_to/related, allConcepts)
     → 更新 store 中该概念的 depends_on/leads_to/related 字段
     → 对比新旧边集，增删边
     → 增量布局
```

### 架构影响

```
改动前:
  onBackgroundDoubleTap → 只做取消选中

改动后:
  onBackgroundDoubleTap → 弹出 AddConceptDialog → addConcept → 增量布局

  ConceptDetailPanel 编辑文档 → [更新关系和图谱] 按钮
    → parseFrontmatter → updateEdges → 增量布局
```

## 影响范围

- `magic-memory/src/components/KnowledgeGraphView.tsx` — 接入 onBackgroundDoubleTap 的添加概念逻辑
- `magic-memory/src/components/KnowledgeGraph.tsx` — 可能需将双击坐标传给回调
- `magic-memory/src/components/ConceptDetailPanel.tsx` — 新增"更新关系和图谱"按钮
- `magic-memory/src/store/knowledgeGraphStore.ts` — 新增 reparseRelations action
- `magic-memory/src/utils/conceptParser.ts` — 增强 matchTitlesToIds 以支持边集 diff
