# 项目化知识库设计文档

## 1. 概述

### 背景
当前知识库系统将所有概念和图谱数据存储在单一全局空间中，用户无法按项目/文件夹隔离和管理不同的知识图谱。

### 目标
支持用户创建多个独立项目，每个项目对应一个文件夹，图谱数据存储在 `~/.magic-memory/projects/{project-id}/` 目录下，实现项目级别的数据隔离。

### 核心价值
- 数据隔离：不同项目的图谱相互独立
- 灵活切换：用户可在项目间快速切换
- 持久存储：基于文件系统，脱离浏览器 localStorage 限制

---

## 2. 需求澄清

| 需求项 | 澄清结果 |
|--------|----------|
| 项目定义 | 用户在 UI 中选择文件夹目录作为项目 |
| 图谱策略 | 每个项目拥有独立的概念和边数据 |
| 存储位置 | `~/.magic-memory/projects/{project-id}/` |
| UI 入口 | 文档视图 → 右侧显示项目列表 + 添加按钮 |

---

## 3. 数据模型

### 3.1 项目结构
```
~/.magic-memory/
├── projects/
│   ├── project-list.json       # 项目元数据列表
│   └── {project-id-1}/
│       ├── config.json         # 项目配置（名称、源文件夹路径）
│       ├── concepts.json       # 概念数据
│       ├── edges.json          # 边数据
│       └── graph-summary.json  # 图谱摘要缓存（可选）
```

### 3.2 TypeScript 类型

```typescript
// 项目元数据
interface Project {
  id: string;                    // 项目唯一 ID (UUID)
  name: string;                  // 项目显示名称（文件夹名）
  folderPath: string;            // 源文件夹绝对路径
  createdAt: string;             // ISO 时间
  lastOpenedAt: string;          // ISO 时间
}

// 项目配置 (config.json)
interface ProjectConfig {
  id: string;
  name: string;
  folderPath: string;
  createdAt: string;
  lastOpenedAt: string;
}

// 扩展知识图谱 Store
interface KnowledgeGraphStore {
  // 新增字段
  projects: Project[];
  currentProjectId: string | null;

  // 新增方法
  addProject: (project: Project) => void;
  removeProject: (projectId: string) => void;
  setCurrentProject: (projectId: string) => void;
  loadProjects: () => Promise<void>;
  loadProjectGraph: (projectId: string) => Promise<void>;
}
```

---

## 4. API 设计

### 4.1 新增 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/projects` | 获取项目列表 |
| POST | `/api/projects` | 创建新项目（扫描文件夹生成图谱） |
| GET | `/api/projects/:id` | 获取项目详情和图谱数据 |
| DELETE | `/api/projects/:id` | 删除项目（删除目录） |
| PUT | `/api/projects/:id` | 更新项目配置（重命名等） |

### 4.2 现有 API 改造

| 方法 | 路径 | 改动 |
|------|------|------|
| GET | `/api/graph` | 新增可选参数 `?projectId=xxx`，默认当前项目 |
| POST | `/api/scan-docs` | 新增可选参数 `?projectId=xxx`，扫描结果写入指定项目 |
| GET | `/api/cluster` | 新增可选参数 `?projectId=xxx` |

### 4.3 请求/响应示例

```bash
# 获取项目列表
GET /api/projects

# 响应
{
  "projects": [
    {
      "id": "proj_abc123",
      "name": "vLLM 学习笔记",
      "folderPath": "/Users/meetai/docs/vllm",
      "createdAt": "2026-05-01T10:00:00Z",
      "lastOpenedAt": "2026-05-01T15:30:00Z"
    }
  ]
}

# 创建新项目
POST /api/projects
Content-Type: application/json

{
  "name": "新项目",
  "folderPath": "/Users/meetai/docs/my-project"
}

# 响应
{
  "project": { /* Project 对象 */ },
  "concepts": [ /* 概念数组 */ ],
  "edges": [ /* 边数组 */ ]
}

# 获取项目图谱
GET /api/graph?projectId=proj_abc123

# 响应
{
  "concepts": [...],
  "edges": [...]
}
```

---

## 5. 前端改动

### 5.1 Store 改动 (`knowledgeGraphStore.ts`)

```typescript
// 新增状态和方法
interface KnowledgeGraphStore {
  projects: Project[];
  currentProjectId: string | null;

  // 方法
  loadProjects: () => Promise<void>;
  createProject: (name: string, folderPath: string) => Promise<Project>;
  deleteProject: (projectId: string) => Promise<void>;
  switchProject: (projectId: string) => Promise<void>;
}
```

### 5.2 UI 改动

#### 5.2.1 文档视图侧边栏改造
```
┌─────────────────────────────────────┐
│ 📁 文档                        [+] │
├─────────────────────────────────────┤
│ │ 项目列表                     │ │
│ │ ├─ 项目 A ✓ (当前)           │ │
│ │ ├─ 项目 B                    │ │
│ │ └─ 项目 C                    │ │
│ │                              │ │
│ │ [+ 添加项目文件夹]           │ │
│ │                              │ │
│ │ 文档列表                     │ │
│ │ ├─ 文档 1                    │ │
│ │ └─ 文档 2                    │ │
└─────────────────────────────────────┘
```

#### 5.2.2 交互流程
1. 用户点击「添加项目文件夹」→ 调用系统文件夹选择器
2. 选择文件夹 → 调用 `POST /api/projects`
3. 服务端创建项目目录、扫描文件、生成图谱 → 返回数据
4. 前端自动切换到新项目，显示其图谱
5. 点击项目列表中的项目 → 切换并加载对应图谱

### 5.3 组件职责

| 组件 | 职责 |
|------|------|
| `Sidebar.tsx` | 保留文档列表，右侧新增项目列表区域 |
| `ProjectList.tsx` (新) | 项目列表展示、添加、删除、切换 |
| `Toolbar.tsx` | 保留现有视图切换，考虑添加当前项目指示器 |
| `KnowledgeGraphView.tsx` | 改造为支持项目级别的图谱加载 |

---

## 6. 服务端实现

### 6.1 项目管理模块

```typescript
// 核心路径
const MAGIC_MEMORY_DIR = expandUser('~/.magic-memory');
const PROJECTS_DIR = join(MAGIC_MEMORY_DIR, 'projects');
const PROJECT_LIST_FILE = join(PROJECTS_DIR, 'project-list.json');

// API 实现要点
async function createProject(name: string, folderPath: string) {
  const projectId = `proj_${Date.now()}_${nanoid(6)}`;
  const projectDir = join(PROJECTS_DIR, projectId);

  // 1. 创建目录
  mkdirSync(projectDir, { recursive: true });

  // 2. 扫描源文件夹
  const { concepts, edges } = await scanDirectory(folderPath);

  // 3. 写入项目数据
  await writeFile(join(projectDir, 'config.json'), JSON.stringify({...}));
  await writeFile(join(projectDir, 'concepts.json'), JSON.stringify(concepts));
  await writeFile(join(projectDir, 'edges.json'), JSON.stringify(edges));

  // 4. 更新项目列表
  await updateProjectList();

  return { projectId, concepts, edges };
}

async function loadProjectGraph(projectId: string) {
  const projectDir = join(PROJECTS_DIR, projectId);
  const concepts = JSON.parse(await readFile(join(projectDir, 'concepts.json')));
  const edges = JSON.parse(await readFile(join(projectDir, 'edges.json')));
  return { concepts, edges };
}

async function deleteProject(projectId: string) {
  const projectDir = join(PROJECTS_DIR, projectId);
  await rm(projectDir, { recursive: true });
  await updateProjectList();
}
```

### 6.2 数据迁移

现有全局图谱数据迁移方案：
- 启动服务时检查 `~/.magic-memory/projects/default/` 是否存在
- 如不存在，将 `data/graph-summary.json` 的数据迁移过去作为默认项目
- 前端首次加载时自动选择 default 项目

---

## 7. 边界情况处理

| 场景 | 处理方式 |
|------|----------|
| 选择的文件夹已被删除 | 显示错误提示，标记项目为「不可用」 |
| 项目文件夹内容变更 | 提供「重新扫描」按钮刷新图谱 |
| 项目名称冲突 | 自动添加数字后缀 |
| 服务端首次启动无项目 | 创建默认空项目，引导用户添加 |

---

## 8. 实施计划

### Phase 1: 服务端基础
- [ ] 创建 `~/.magic-memory/projects/` 目录结构
- [ ] 实现项目 CRUD API
- [ ] 实现项目图谱加载/保存
- [ ] 数据迁移脚本

### Phase 2: 前端 Store
- [ ] 扩展 `knowledgeGraphStore.ts`
- [ ] 实现项目列表加载
- [ ] 实现项目切换逻辑
- [ ] API 调用适配

### Phase 3: UI 实现
- [ ] 项目列表组件 (`ProjectList.tsx`)
- [ ] 侧边栏改造（项目 + 文档双栏）
- [ ] 添加项目文件夹交互
- [ ] 项目删除确认

### Phase 4: 收尾
- [ ] 图谱视图适配项目切换
- [ ] 测试多项目切换
- [ ] 文档更新

---

## 9. 验收标准

- [ ] 用户可通过文件夹选择器创建项目
- [ ] 每个项目的图谱数据存储在独立目录
- [ ] 切换项目时图谱自动更新
- [ ] 删除项目时数据一并清除
- [ ] 服务重启后项目列表持久化
- [ ] 现有全局图谱数据可正常迁移

---

## 10. 风险与限制

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 文件夹路径变更 | 项目无法加载 | 记录原始路径，提示用户重新选择 |
| 并发写入冲突 | 数据损坏 | 单线程处理，必要时加锁 |
| 大文件夹扫描慢 | 阻塞响应 | 考虑后台处理，返回进度 |

---

## 11. 附录

### A. 文件路径参考
- 服务端入口: `server.ts`
- 前端 Store: `src/store/knowledgeGraphStore.ts`
- 侧边栏组件: `src/components/Sidebar.tsx`
- 类型定义: `src/types/index.ts`

### B. 相关脚本
- 目录扫描: `src/utils/fileSystem.ts`
- 概念解析: `src/utils/conceptParser.ts`
- 聚类管道: `scripts/cluster.ts`