# Proposal: 基于 FileSystemDirectoryHandle 的多项目管理

## 现状问题

1. **projectStore.createProject 需要文件系统路径** — 浏览器 `showDirectoryPicker` 返回的 `FileSystemDirectoryHandle` 不暴露完整路径，导致需要用户手动填路径，体验极差
2. **多条路线割裂** — `handleAutoScan`（前端 handle 读文件）、`ClusterView`（后端路径读文件）、`/api/graph/summary`（后端路径）各走各的
3. **localStorage 配额限制** — 概念和边数据量大的时候报 `QuotaExceededError`（~5MB 上限）
4. **切换项目成本高** — 项目切换时需要重新从后端路径加载数据

## 目标

- 去掉 `createProject` 的 `folderPath` 参数，改用 `FileSystemDirectoryHandle`
- 前端统一用 handle 读文件、解析概念/边
- 聚类改用前端 Louvain 库，不再依赖后端路径
- 数据存储从 localStorage 迁移到 IndexedDB（结构化克隆支持存 handle）
- 支持多项目无缝切换

## 非目标

- 不改造 `server.ts` 现有的 `/api/graph`、`/api/graph/summary` 等接口（向后兼容）
- 不做完整的 IndexedDB 持久化方案（先把 handle 存明白）

## 约束

- `FileSystemDirectoryHandle` 序列化后存 IndexedDB（不能用 JSON.stringify，支持 structured clone）
- 跨 session 权限可能丢失，需要 `queryPermission` + `requestPermission`
- 与现有的 `handleAutoScan` / `readMdFiles` 实现复用
