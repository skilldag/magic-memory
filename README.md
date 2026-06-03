# Magic Memory

> 基于数字锚点与问题驱动推导的 vLLM 架构学习系统 — 附带交互式知识图谱 Web 应用

---

## 项目概述

Magic Memory 是一个专为 **vLLM 架构学习**设计的记忆与理解系统，结合了两种互补的学习方法和一个交互式知识图谱 Web 应用：

| 方法 | 解决什么问题 | 文件 |
|------|-------------|------|
| **数字锚点记忆法** | 快速记忆 50+ vLLM 概念的名称和位置 | [memory.md](./memory.md) |
| **问题驱动推导法** | 深入理解概念的设计原因和内部原理 | [design.md](./design.md) |
| **交互式知识图谱** | 可视化概念关系，拖拽推导，标注复盘 | Web UI (`http://localhost:3000`) |

---

## 功能全景

### 🧠 数字锚点记忆系统

将 50 个 vLLM 核心概念对应到 0-50 的数字锚点上，通过"数字 → 锚点图像 → 概念"的三步联想进行记忆。

```
示例：数字 34 → 想象"34 号仓库"(Continuous 谐音) → Continuous Batching
```

- **三级难度递进**：Level 1 (0-9 基础) → Level 2 (10-29 核心) → Level 3 (30-50 高级)
- **概念分类组织**：8 大类（Foundation / Model / Attention / Performance / Scheduling / Serving / Advanced / Infrastructure）
- **完整数据流链路**：从 API 请求到 Token 输出的完整路径记忆
- **自测验证**：每 Level 配套自测题，双向回忆（数字→概念 & 概念→数字）

### 🔍 问题驱动推导学习

通过"过程描述 → 发现矛盾 → 形成问题 → 推导方案 → 揭晓概念"的认知链条深入理解。概念不是知识原子，而是**问题的命名方案**。

- **过程画板 (ProcessCanvas)**：ReactFlow 拖拽搭建概念推导流程，支持自适应布局
- **对照验证 (ComparisonPanel)**：用户推导结果 vs 参考流程自动对比，定位知识缺口
- **渐进式提示**：回忆困难时逐级提供线索，自动记录薄弱环节

### 🕸️ 交互式知识图谱

50+ 概念节点的可视化关系网络，支持多维度交互：

| 交互 | 行为 |
|------|------|
| 单击节点 | 查看概念详情、关联文档、依赖链路 |
| 双击节点 | 进入过程画板，拖拽搭建推导流程 |
| 悬停节点 | 预览概念摘要 |
| 拖拽连线 | 手动建立概念关系 |
| 右键菜单 | 快捷操作（建链、探索、聚焦） |

附加工具：
- **DependencyChainSVG** — 自动生成概念依赖关系的 SVG 流程图
- **Focus/Unfocus** — 聚焦单个概念，隐藏无关节点，减少认知负荷
- **Smart Layout** — 一键自动排列图谱布局

### 🧩 概念聚类与社区发现

基于 **Louvain 社区检测算法**（纯 TypeScript 实现），自动将 50+ 概念聚类为若干个**知识社区**：

```bash
# CLI 方式运行聚类
bun scripts/cluster.ts --dir ../docs --resolution 0.5
```

- **聚类结果可视化**：在 `ClusterView` 中以社区卡片展示，同色节点同社区
- **自动建边**：交叉引用 → 强边(0.9)，同目录 → 弱边(0.2)，编号邻近 → 中边(0.3)
- **Cohesion 指标**：展示每个社区的内聚度，判断概念分组的合理性

### 🔎 全局搜索与探索

| 功能 | 说明 |
|------|------|
| **GlobalSearch** | 全项目概念/文档一键搜索，快速定位 |
| **QuickExploreDialog** | 选中概念后一键探索关联概念 |
| **ExploreDialog** | 深度探索模式，展示概念的全链路关系 |
| **BatchLinkDialog** | AI 推荐潜在关联 + 批量建链 |

### 📚 Web 标注阅读器

基于 Bun + React 的 Web 应用，用于浏览和标注 vLLM 知识文档：

- **文档浏览**：支持 Markdown 渲染（marked + kaTeX + highlight.js + mermaid），按分类筛选
- **注释系统**：支持评论、问题、建议、纠正四种类型，SelectionPopover 行内触发
- **AnnotationPreview**：光标悬停预览注释内容
- **导入导出**：JSON / Markdown / HTML 三种格式导出

### 📊 分析面板

| 面板 | 功能 |
|------|------|
| **AnalysisPanel** | 图谱统计学分析（节点度、社区分布、关键路径） |
| **AlignmentPanel** | 概念对齐展示，对比不同概念在同一维度上的位置 |
| **SummaryPanel** | 学习进度概览，薄弱环节标记，复习提醒 |

### 🧪 测试体系

```bash
# 运行测试
npx vitest run
```

包含 5 个测试文件覆盖核心算法：

| 测试文件 | 覆盖内容 |
|---------|---------|
| `adaptiveLayout.test.ts` | 自适应布局算法 |
| `alignment.test.ts` | 概念对齐逻辑 |
| `compare-methods.test.ts` | 推导结果对比方法 |
| `deriveEdges.test.ts` | 边推导（引用、同目录、编号邻近） |
| `fileSystem.test.ts` | 文件扫描与概念提取 |

---

## 快速开始

### 1️⃣ 安装与启动

```bash
# 安装依赖
npm install

# 启动 Web UI（开发模式）
npm run dev

# 访问 http://localhost:3000
```

### 2️⃣ 使用记忆系统

```bash
# 直接从记忆参考开始
open memory.md

# 按 Level 渐进学习
# Level 1 → docs/Foundation/
# Level 2 → docs/Model/ + docs/Performance/
# Level 3 → docs/Scheduling/ + docs/Serving/

# 自测验证
# docs/Level1-自测验证.md
# docs/Level2-自测验证.md
# docs/Level3-自测验证.md
```

### 3️⃣ 使用 CLI 工具

全局服务 + 项目管理：

```bash
# 启动全局服务和 Web UI（一键）
bun run scripts/memo.ts server start

# 扫描文档目录，构建知识图谱并注册为项目
bun run scripts/memo.ts init ./docs

# 列出已注册项目
bun run scripts/memo.ts list

# 查看服务状态
bun run scripts/memo.ts server status

# 停止服务
bun run scripts/memo.ts server stop
```

CLI 也可全局安装使用：`npm link` 后直接运行 `memo`。

### 4️⃣ 概念聚类

```bash
# 运行 Louvain 社区检测，输出 JSON 供前端渲染
bun run scripts/cluster.ts --dir ./docs --output cluster-result.json

# 指定分辨率（值越小社区越少）
bun run scripts/cluster.ts --dir ./docs --resolution 0.8
```

---

## 项目结构

```
magic-memory/
├── memory.md                     # 数字锚点记忆法（核心记忆参考）
├── design.md                     # 问题驱动推导设计文档
├── docs/                         # vLLM 概念文档库
│   ├── Foundation/               # Level 1: 基础设施
│   ├── Model/                    # Level 2: 模型执行
│   ├── Performance/              # Level 2: 性能优化
│   ├── Scheduling/               # Level 3: 调度
│   ├── Serving/                  # Level 3: 服务化
│   ├── Optimization/             # Level 3: 高级优化
│   ├── Infrastructure/           # Level 3: 基础设施
│   ├── Advanced/                 # Level 3: 高级特性
│   └── superpowers/              # 功能迭代 spec & plan
├── src/                          # React 前端源码
│   ├── components/               # 29 个 React 组件
│   │   ├── KnowledgeGraph.tsx    # 知识图谱主组件
│   │   ├── KnowledgeGraphView.tsx
│   │   ├── ClusterView.tsx       # 社区聚类视图
│   │   ├── ProcessCanvas.tsx     # 推导画板 (ReactFlow)
│   │   ├── ComparisonPanel.tsx   # 对照验证
│   │   ├── AlignmentPanel.tsx    # 概念对齐
│   │   ├── AnalysisPanel.tsx     # 图谱分析
│   │   ├── DocumentViewer.tsx    # 文档阅读器
│   │   ├── AnnotationPanel.tsx   # 注释面板
│   │   ├── AnnotationDialog.tsx  # 注释编辑器
│   │   ├── AnnotationPreview.tsx # 注释预览
│   │   ├── GlobalSearch.tsx      # 全局搜索
│   │   ├── ExploreDialog.tsx     # 概念探索
│   │   ├── BatchLinkDialog.tsx   # 批量建链
│   │   ├── ExportModal.tsx       # 导出
│   │   ├── ImportModal.tsx       # 导入
│   │   ├── Sidebar.tsx           # 侧边栏
│   │   ├── Toolbar.tsx           # 工具栏
│   │   ├── Toast.tsx             # 消息提示
│   │   └── ...                   # 更多组件
│   ├── store/                    # Zustand 状态管理
│   │   ├── knowledgeGraphStore.ts # 知识图谱状态
│   │   ├── documentStore.ts      # 文档状态
│   │   ├── annotationStore.ts    # 注释状态
│   │   ├── projectStore.ts       # 项目管理
│   │   └── toastStore.ts         # 消息提示
│   ├── hooks/                    # 自定义 Hooks
│   ├── types/                    # TypeScript 类型定义
│   ├── utils/                    # 工具函数
│   └── workers/                  # Web Workers
├── server/                       # 后端服务
│   ├── explore.ts                # 全局 API 服务（端口 4321）
│   └── graphBuilder.ts           # 图谱构建引擎
├── scripts/                      # CLI 工具
│   ├── memo.ts                   # 项目管理 CLI（init/list/server）
│   └── cluster.ts                # Louvain 社区检测管道
├── tests/                        # 测试套件
│   ├── adaptiveLayout.test.ts
│   ├── alignment.test.ts
│   ├── compare-methods.test.ts
│   ├── deriveEdges.test.ts
│   └── fileSystem.test.ts
└── data/                         # 运行时数据
```

---

## 学习路径建议

```
第1周: Level 1 (0-9) 基础设施
  → 记忆锚点 → 阅读 Foundation 文档 → 自测验证
  → Web UI 查看概念关系图

第2周: Level 2 (10-29) 模型执行
  → Model 加载流程 → PagedAttention → Sampler → Forward
  → 在 ProcessCanvas 拖拽推导 Attention 流程
  → 用 ComparisonPanel 对照验证

第3周: Level 3 (30-50) 高级特性
  → Speculative Decoding → Scheduling → Serving
  → ClusterView 查看社区聚类 → 理解跨概念关联

第4周: 整合复习
  → DependencyChainSVG 默写数据流路径
  → AlignmentPanel 对比相似概念
  → 定期复盘: docs/定期复盘机制.md
```

---

## 相关文档

| 文档 | 说明 |
|------|------|
| [memory.md](./memory.md) | 数字锚点记忆法 — 快速记忆 50+ 概念 |
| [design.md](./design.md) | 问题驱动推导 — 深入理解设计原理 |
| [知识体系化框架](./docs/知识体系化框架.md) | 概念之间的关系和组织方式 |
| [vLLM概念推演过程](./docs/vLLM概念推演过程.md) | 从问题到概念的完整推演 |
| [定期复盘机制](./docs/定期复盘机制.md) | 间隔复习和巩固计划 |

---

## 开发指引

### 常用命令

```bash
npm run dev          # 启动开发服务器
npm run build        # 生产构建
npm run typecheck    # TypeScript 类型检查
npm test             # 运行测试
bun run scripts/memo.ts server start  # 启动完整服务
```

### 技术栈

| 层级 | 技术 |
|------|------|
| **前端框架** | React 19 + TypeScript + Vite 6 |
| **样式** | Tailwind CSS 4 |
| **状态管理** | Zustand 5 + Immer |
| **流程图** | @xyflow/react (ReactFlow) |
| **图算法** | graphology + graphology-communities-louvain |
| **Markdown** | marked + kaTeX + highlight.js + mermaid + DOMPurify |
| **可视化** | @viz-js/viz (Graphviz) |
| **AI/ML** | @xenova/transformers (浏览器端 Embedding) |
| **后端** | Bun + TypeScript |
| **测试** | Vitest |
| **中文 NLP** | jieba (分词) |
| **工具** | diff, overlayscrollbars |

### 架构说明

- **前端**：Vite SPA，所有图计算在浏览器端完成（Louvain 聚类、布局计算）
- **后端**：Bun HTTP 服务，负责项目注册、图谱持久化、文件扫描
- **CLI**：Bun 脚本，提供项目管理/聚类/服务控制能力
- **数据流**：文档目录 → 扫描提取概念 → 构建图（交叉引用/目录/编号）→ Louvain 社区检测 → 前端可视化

---

## Feature 开发工作流

本项目的功能迭代使用 **OpenSpec** 管理，所有变更记录在 `openspec/` 和 `docs/superpowers/` 目录下：

```
docs/superpowers/
├── specs/       # 功能设计文档
└── plans/       # 实现计划
```

每个功能遵循：**设计文档 → 实现计划 → 编码 → 测试 → 复盘** 的完整流程。
