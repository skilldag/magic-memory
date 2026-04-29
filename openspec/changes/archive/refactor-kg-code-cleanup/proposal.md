# 知识图谱代码重构

## 问题

- KnowledgeGraphView.tsx 1019 行，混杂图谱编排、8 个 tab 内容、2 个弹窗、SVG、纯函数
- ExploreDialog 边创建逻辑重复 2 次
- 样式常量散落在组件中无法复用
- getNameReason 函数定义了但从未使用（死代码）

## 目标

| 文件 | 当前 | 目标 |
|------|------|------|
| KnowledgeGraphView.tsx | 1019 行 | ~150 行 |
| 内联弹窗 | 2 个 | 独立组件 |
| 纯函数 | 在组件内 | utils/knowledgeGraph.ts |
| 样式常量 | 在 KG.tsx | constants/graph.ts |
| 重复边创建 | 2 处 | 0 处 |
