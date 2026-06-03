# Magic Memory

> AI 驱动的知识图谱学习系统 — 自动生成概念节点，对齐理解度评估，SM-2 间隔重复复习

---

## 项目概述

Magic Memory 是一个**交互式知识图谱学习系统**，围绕三个核心环节构建学习闭环：

```
  📥 AI 从文档提取/生成概念节点   →   🧩 知识图谱可视化
              ↓                                ↓
  📊 用户自述 → KEY CONCEPTS 对齐评估   →   🔁 SM-2 间隔复习
```

- **🤖 AI 生成节点**：从文档 `KEY CONCEPTS` 解析概念，AI 推荐提炼新概念并自动建链
- **📊 对齐评估**：用户自述 vs 原文 KEY CONCEPTS 自动对比，计算覆盖率和掌握度
- **🔁 SM-2 复习**：每次对齐完成后触发 SM-2 排期，图谱显示复习徽章和掌握度颜色

---

## 功能全景

### 🧩 交互式知识图谱

由 **Cytoscape.js** 渲染的关系图谱，支持全维度交互：

| 交互 | 行为 |
|------|------|
| 单击节点 | 查看概念详情、关联文档、对齐面板 |
| 右键菜单 | 手动建链、AI 探索、聚焦视图 |
| 悬停节点 | 预览概念摘要 |
| 拖拽连线 | 连线模式下手动建立关系 |
| 图例 | 关系类型（依赖/引出/相关）+ 掌握度色阶 |

图谱控制：

- **全局搜索** — 输入概念名即时定位
- **自适应布局** — 一键自动排列
- **聚焦模式 (Focus View)** — BFS 展开关联层，深度可调 (1-3 层)
- **焦点链路着色** — 黄色入边 / 绿色出边，直观展示流向
- **复习模式** — 切换后图谱显示待复习节点徽章

### 🤖 AI 概念生成与建链

| 功能 | 说明 |
|------|------|
| **KEY CONCEPTS 解析** | 从文档中 `# KEY CONCEPTS` 章节自动提取概念创建节点 |
| **AI 建议建链** | 选中节点后 AI 分析 `problem` 字段，推荐关联概念 |
| **一键批量建链** | `BatchLinkDialog` 展示 AI 推荐列表，勾选即建 |
| **探索模式** | `ExploreDialog` — 从节点出发，AI 生成延伸问题并提炼为新概念 |
| **快捷探索** | `QuickExploreDialog` — 选中文本一键创建节点并关联 |
| **连接模式** | 工具栏切换连线模式，手动拖拽建链 |

### 📊 理解度对齐评估 (Alignment)

**核心机制** — 用户阅读后用自己的话描述，系统自动对比原文：

```
  原文 KEY CONCEPTS              用户自述             对齐结果
  ┌─────────────────┐         ┌──────────────┐      ✅ PagedAttention
  │ PagedAttention   │         │ "Attention    │      ❌ Block Table
  │ Block Table      │ ────→   │  通过 Q/K/V   │      ⚠️ Q/K/V (模糊匹配)
  │ CacheBlock       │         │   计算权重..." │
  └─────────────────┘         └──────────────┘
```

- **图谱级对比**：同时对比节点（概念覆盖）和边（关系理解）
- **三级匹配策略**：精确子串 → charJaccard 模糊 → **transformers.js 语义匹配**
- **浏览器端语义模型**：`all-MiniLM-L6-v2`，无后端依赖
- **手动干预**：标记已理解 / 忽略 / 从原文删除
- **掌握度评分**：综合对齐次数、覆盖率、手动标记，0-100 分
- **颜色反馈**：图谱节点背景色随掌握度变化（灰→黄→绿→蓝）

### 🔁 SM-2 间隔复习系统

每次对齐完成后自动触发：

```
  用户完成对齐
      ↓
  评估理解质量 (0-5) ──→ SM-2 算法 ──→ 更新 ease_factor
                                       计算 interval
                                       设定 next_review
                                           ↓
  图谱显示复习徽章 ←───────────── 到期概念入队
  SummaryPanel 复习队列             横幅提醒
```

- **复习徽章**：🔥(逾期) / 今日 / N天 / New / ✓
- **复习队列**：`SummaryPanel` 按 urgency 排序展示待复习列表
- **提醒横幅**：待复习数量和最久逾期天数
- **掌握度色阶**：图谱颜色随复习进度变化

### 📚 文档阅读与标注

- **Markdown 渲染**：kaTeX 公式 + highlight.js 代码高亮 + mermaid 图表
- **注释系统**：评论 / 问题 / 建议 / 纠正，行内触发 + 悬停预览
- **分类筛选**：按 Foundation / Model / Attention 等分类浏览

### 🧭 分析面板

| 面板 | 功能 |
|------|------|
| **SummaryPanel** | 图谱统计、入口节点、枢纽、最长路径、待复习队列、复习准时率 |
| **AnalysisPanel** | 数据流路径展示与追踪 |
| **AlignmentPanel** | 当前概念对齐结果详情 |
| **ClusterView** | Louvain 社区聚类可视化 |

### 📦 多项目管理 CLI

```bash
memo init ./docs      # 扫描目录构建图谱
memo list             # 列出项目
memo remove <id>      # 删除项目
memo server start     # 启动服务
memo server stop      # 停止服务
```

---

## 架构

```
                  ┌──────────────────────┐
  用户自述 ──────→│  Vite SPA (端口 3000)  │────→ 浏览器端:
  文档选择        │  React 19             │     - 图谱渲染 (Cytoscape)
                  │  Zustand 状态管理       │     - Louvain 聚类
                  │  transformers.js 语义   │     - SM-2 调度
                  └──────────┬───────────┘     - KEY CONCEPTS 对齐
                             │ API
                             ↓
                  ┌──────────────────────┐
                  │  Bun HTTP (端口 4321)  │
                  │  项目注册 / 图谱持久化   │
                  │  文件扫描              │
                  └──────────────────────┘
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| **前端** | React 19 + TypeScript + Vite 6 + Tailwind CSS 4 |
| **状态** | Zustand 5 + Immer |
| **图谱** | Cytoscape.js (`@viz-js/viz`) |
| **图算法** | graphology + Louvain |
| **语义匹配** | `@xenova/transformers` (all-MiniLM-L6-v2, 浏览器端) |
| **间隔复习** | SM-2 算法（纯前端） |
| **流程图** | `@xyflow/react` (ReactFlow) |
| **文档渲染** | marked + kaTeX + highlight.js + mermaid |
| **后端** | Bun + TypeScript |
| **测试** | Vitest |
| **NLP** | jieba |

---

## 快速开始

```bash
npm install
npm run dev            # http://localhost:3000

# 或一键启动（需要 Bun）
bun run scripts/memo.ts server start
```

首次使用，点 **"添加项目"** 选择 Markdown 文档目录，系统自动构建图谱。

### 学习流程

```
阅读文档 → 用自己的话描述 → 对齐评估 → 接收复习提醒 → 间隔复习
```

---

## 项目结构

```
magic-memory/
├── src/
│   ├── components/       # 27 个组件
│   │   ├── KnowledgeGraph.tsx    # 图谱主组件 (Cytoscape)
│   │   ├── AlignmentPanel.tsx    # KEY CONCEPTS 对齐
│   │   ├── SummaryPanel.tsx      # 摘要 + 复习队列
│   │   ├── ExploreDialog.tsx     # AI 探索
│   │   ├── BatchLinkDialog.tsx   # 批量建链
│   │   └── ...
│   ├── store/            # Zustand 状态
│   │   └── knowledgeGraphStore.ts  # 核心: 概念/掌握度/复习/对齐
│   ├── utils/            # 核心算法
│   │   ├── alignment.ts         # KEY CONCEPTS 对齐
│   │   ├── semanticMatch.ts     # transformers.js 语义匹配
│   │   ├── knowledgeGraph.ts    # SM-2 调度 + 复习徽章
│   │   └── graphAnalysis.ts     # 图谱分析
│   └── types/
├── server/               # Bun API (端口 4321)
├── scripts/              # CLI: memo / cluster
├── tests/                # Vitest
└── docs/                 # 知识文档
```

---

## 学习路径

1. **导入项目** — 选择文档目录，自动构建知识图谱
2. **浏览图谱** — 三种视图切换，了解概念全貌
3. **逐概念学习** — 单击节点 → 阅读文档 → 对齐评估
4. **间隔复习** — 跟随图谱徽章和复习队列定期回顾
5. **AI 扩展** — 使用探索/建链功能扩展图谱
