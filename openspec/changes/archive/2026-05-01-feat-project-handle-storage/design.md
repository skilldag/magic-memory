# Design: 基于 FileSystemDirectoryHandle 的多项目管理

## 架构

```
┌─────────────────────────────────────────────────────┐
│  Sidebar                                            │
│  ┌──────────────────────────────────────────┐       │
│  │  ProjectList                              │       │
│  │  ┌─────┐ ┌─────┐ ┌─────┐                 │       │
│  │  │项目A│ │项目B│ │ +  │                 │       │
│  │  └─────┘ └─────┘ └─────┘                 │       │
│  └──────────────────────────────────────────┘       │
│  ── 项目切换 ──────────────────────────────────       │
│  文档列表/概念/图谱                                        │
└─────────────────────────────────────────────────────┘
         │ 选文件夹
         ▼
┌──────────────────────┐    structured clone    ┌──────────────────┐
│ showDirectoryPicker  │ ──────────────────────→ │    IndexedDB     │
│ 返回 handle          │    存 handle+元数据     │  projectHandles  │
└──────────────────────┘                        └──────────────────┘
         │                                              │
         ▼ 读文件                                       │ 切项目时读取
┌──────────────────────┐                               │
│  readMdFiles(handle) │ ←──────────────────────────────┘
│  parseFrontmatter    │    取出 handle → 请求权限
│  概念+边             │    → readMdFiles → 解析
└──────────────────────┘
         │
         ▼ 存到内存
┌──────────────────────┐
│  knowledgeGraphStore │
│  concepts / edges    │ ← 也是 Louvain 聚类的输入
└──────────────────────┘
         │
         ▼
┌──────────────────────┐
│  ClusterView         │
│  前端 Louvain 聚类    │ ─→ 显示社区/路径
│  (graphology-louvain)│
└──────────────────────┘
```

## 数据流

### 创建项目
1. 用户点击 "+" → `showDirectoryPicker()` → `handle`
2. `handle.name` 作为项目名
3. `handle` + 项目元数据 → 存 IndexedDB（key: projectId）
4. `readMdFiles(handle)` → 解析 frontmatter → concepts/edges
5. 概念/边 → `knowledgeGraphStore.setState()`

### 切换项目
1. 从 IndexedDB 取出对应 handle
2. `handle.queryPermission()` — 如果被拒绝，调 `requestPermission()` 弹授权
3. 授权通过 → `readMdFiles(handle)` → 解析
4. 概念/边 → `knowledgeGraphStore.setState()`
5. 加载状态显示 loading

### 聚类（ClusterView）
1. 从 `knowledgeGraphStore` 取当前项目的 concepts/edges
2. 用 `graphology` 构建图 + `graphology-louvain` 跑社区检测
3. 结果渲染（同现有 ClusterView UI）

## 涉及改动

### 新增
- `src/utils/handleStorage.ts` — IndexedDB 存取 handle 的工具函数

### 修改
- `src/store/projectStore.ts` — `createProject` 去掉 `folderPath`，改存 handle
- `src/types/project.ts` — `Project` 类型增加 `handleHandleId` 字段（指向 IndexedDB 中的 handle）
- `src/types/index.ts` — 恢复 `export type { Project } from './project'`
- `src/components/Sidebar.tsx` — `handleAddProject` 去掉 prompt，直接存 handle
- `src/components/ClusterView.tsx` — 前端 Louvain 替代后端 API 调用
- `src/components/KnowledgeGraphView.tsx` — `handleBrowseFolder` 复用 handleStorage

### 依赖
- 安装 `graphology` + `graphology-louvain`（前端聚类）

## 边界情况

| 场景 | 处理 |
|------|------|
| 用户拒绝权限 | 显示提示，停留在当前项目 |
| handle 过期/失效 | 清除该项目，提示重新选择文件夹 |
| IndexedDB 写入失败 | 降级到 localStorage 存元数据，handle 不持久化 |
| 空文件夹 | 显示"未发现概念文档"，不切换项目 |
| 大项目读文件慢 | Button 加 loading 态，禁止重复点击 |
