# Proposal: 知识图谱索引生成 — 自动/手动双模式

## 现状

目前知识图谱索引完全依赖 `mockGraphData.ts` 中 51 个硬编码的概念数据：

```typescript
// mockGraphData.ts
export const mockConcepts: Concept[] = [
  { id: '0', title: 'VllmConfig', content: `...`, path: '...', ... },
  ...
]
```

用户新增的概念只在内存中存在，刷新页面后被 mock 数据覆盖。虽然最新代码已将 concepts/edges 加入 Zustand persist，但数据源仍然是 mock 硬编码。

## 问题

1. **不灵活** — 改图谱必须改代码，无法通过操作文档来更新图谱
2. **没有增长性** — 用户新增概念需手动编辑代码或依赖 AI 对话框，操作链路长
3. **mock 数据是黑盒** — docs/ 目录中的真实文档和 mock 数据是两张皮，容易不一致
4. **无自动导入能力** — 已有知识库文档无法批量导入图谱

## 方案

### 两种索引生成模式

| 模式 | 触发方式 | 数据源 | 用途 |
|------|---------|--------|------|
| **自动扫描** | 用户输入路径 → 开始扫描 | docs/ 目录 .md 文件 | 批量导入，适合首次建索引 |
| **手动添加** | 用户输入概念名 | LLM 分析生成 | 日常增长，逐步完善图谱 |

### 架构变化

```
之前: mockGraphData.ts → loadGraph() → 图谱
之后: localStorage (索引) → 图谱
       ↑                    ↑
   自动扫描/manual      path → docLoader → 文档
```

### 关键改动

1. 添加 `/api/scan-docs` 后端接口（遍历目录 + 解析 frontmatter）
2. 添加 `/api/manual-add-concept` 后端接口（LLM 分析生成概念）
3. 移除 `mockGraphData.ts` 作为默认数据源
4. 前端启动时检查 localStorage，无数据则显示空白引导
5. 文档缺失时显示 [请求 LLM 生成] 按钮

## 影响范围

- `magic-memory/src/store/knowledgeGraphStore.ts` — loadGraph 逻辑
- `magic-memory/src/components/KnowledgeGraphView.tsx` — 空白引导 UI
- `magic-memory/src/components/ConceptDetailPanel.tsx` — 文档缺失 LLM 按钮
- `magic-memory/server.ts` — 新增 /api/scan-docs 等接口
- `magic-memory/src/utils/conceptParser.ts` — 增强 frontmatter 解析
