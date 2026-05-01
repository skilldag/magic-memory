# Tasks: feat-project-handle-storage

## T1: handleStorage 工具函数
- [ ] 创建 `src/utils/handleStorage.ts`
  - [ ] `saveHandle(id, handle)` — handle 存 IndexedDB（structured clone）
  - [ ] `loadHandle(id)` — 从 IndexedDB 取 handle
  - [ ] `deleteHandle(id)` — 删除 handle
  - [ ] `ensurePermission(handle)` — queryPermission + requestPermission 封装
- [ ] 安装 `idb`（IndexedDB 封装库，可选，也可直接用原生 API）

## T2: Project 类型改造
- [ ] `src/types/project.ts` — `Project` 接口增加 `handleStoreId: string | null`
- [ ] `src/types/index.ts` — 恢复 `export type { Project } from './project'`
- [ ] `src/store/projectStore.ts`
  - [ ] `createProject(name, folderPath)` → `createProject(name)`，内部调 `showDirectoryPicker`
  - [ ] 创建项目时存 handle 到 IndexedDB，handleStoreId 存 project 对象
  - [ ] `switchProject` 改为从 handle 读文件而非调后端 API
  - [ ] 去掉 `folderPath` 相关逻辑

## T3: Sidebar 改造
- [ ] `handleAddProject`：
  - [ ] 直接 `showDirectoryPicker()` → handle
  - [ ] 提示项目名称（默认 handle.name）
  - [ ] `createProject(name)` 传 handle
  - [ ] 去掉旧的 `prompt('请输入路径')` 逻辑

## T4: 前端 Louvain 聚类
- [ ] 安装 `graphology` + `graphology-louvain` + `graphology-library`
- [ ] `src/components/ClusterView.tsx`：
  - [ ] 不再调 `/api/cluster?path=...`
  - [ ] 从 `useKnowledgeGraphStore` 取 concepts/edges
  - [ ] 用 `graphology` 构建 Graph 对象
  - [ ] 用 `graphology-louvain` 跑社区检测
  - [ ] 转换为现有的 Community/ClusterResult 格式
  - [ ] 保持现有 UI（社区颜色、内聚度条等）不变

## T5: 边界处理
- [ ] 权限拒绝时显示提示，不崩溃
- [ ] handle 失效时清除项目条目，提示重新选文件夹
- [ ] 空文件夹提示
- [ ] 项目切换 loading 状态

## T6: 归档旧 change
- [ ] 确认所有改动通过 build
- [ ] `openspec archive feat-project-handle-storage`
