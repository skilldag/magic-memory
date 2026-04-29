# 重构设计

## 拆分结构

KnowledgeGraphView.tsx (1019行)
├── ConceptDetailPanel.tsx     → 右侧面板 + 8 tab 内容
├── ManualAddDialog.tsx        → 手动添加弹窗
├── BatchLinkDialog.tsx        → AI 批量添加弹窗
├── DependencyChainSVG.tsx     → 推导路径 SVG
├── utils/knowledgeGraph.ts    → 6 个纯函数
└── constants/graph.ts         → LEVEL_COLORS, EDGE_COLORS

## 其他变更

- knowledgeGraphStore.ts → add createConceptWithEdges
- ExploreDialog.tsx → 复用 createConceptWithEdges 去重
- KnowledgeGraph.tsx → 导入 constants 替代内联定义
