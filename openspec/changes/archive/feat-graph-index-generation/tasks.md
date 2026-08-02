# Tasks: feat-graph-index-generation

> 收尾说明：本 change 的核心目标（取代 mock 硬编码索引）已由 CLI `memo init-repo` + `server/graphBuilder.ts` 落地。
> T2 `/api/scan-docs` 的扫描逻辑保留在 `buildGraphFromDir()`（入口改为 CLI），T4 `/api/manual-add-concept` 不实现
> （前端手动添加改走 QuickExploreDialog → /api/explore 的 LLM 流程）。

## T1: 添加空白图谱引导 UI
- [x] KnowledgeGraphView 中检测 concepts 为空时，显示设置路径/选择模式界面
- [x] 输入 docs 路径的 input + [自动扫描] [手动添加] 两个按钮
- [x] localStorage 有数据时跳过引导，正常渲染图谱
  - 注：引导 UI 保留（showProjectList），交互从「UI 输入路径」演进为「CLI 命令提示 + 已注册项目列表」

## T2: 实现后端 /api/scan-docs 接口
- [x] server.ts 新增 POST /api/scan-docs { path }
  - 注：HTTP 接口已移除；等价逻辑 `buildGraphFromDir()`（graphBuilder.ts）保留，入口为 CLI `memo init-repo`
- [x] 递归遍历目录下所有 .md 文件
- [x] 解析 frontmatter → ConceptIndex
- [x] 无 frontmatter 的文件调 LLM 分析内容→生成 frontmatter
- [x] 构建 edges，返回 { concepts, edges }

## T3: 增强 conceptParser.ts
- [x] parseFrontmatter 函数复用现有实现
  - 注：parseFrontmatter 实现在 server/graphBuilder.ts:98，被 buildGraphFromDir 使用
- [x] 添加 title-to-id 模糊匹配函数
  - 注：matchTitlesToIds 实现在 server/graphBuilder.ts:197
- [x] 添加从文件路径推断 category 的函数
  - 注：category 由 frontmatter 提供，缺失时回退为空字符串

## T4: 实现后端 /api/manual-add-concept 接口
- [ ] server.ts 新增 POST /api/manual-add-concept { title, context }
  - 不实现：前端手动添加已走 QuickExploreDialog → POST /api/explore 的 LLM 结构化概念生成流程
- [ ] LLM 分析返回结构化概念数据
  - 不实现：同上
- [ ] 标题匹配到已有概念 ID
  - 不实现：同上

## T5: 前端手动添加流程
- [x] 增强现有的 / 按钮（QuickExploreDialog），输入概念名
- [x] 调用 LLM → 接收结构化数据 → 匹配 → 创建索引
- [x] 写入 .md 文件
- [x] 更新 localStorage → 图谱刷新
  - 注：持久化已从 localStorage 迁移到项目目录 concepts.json/edges.json

## T6: 实现文档缺失 LLM 生成
- [x] ConceptDetailPanel 中 docContent 为 null 时显示 LLM 按钮
- [x] 调用 /api/generate-doc 生成全文
- [x] 写入 {docs_path}/{path}
- [x] 重新加载显示

## T7: 移除 mockGraphData.ts 依赖
- [x] loadGraph() 不再从 mockGraphData 初始化
- [x] 删除 mockConcepts、mockEdges、mockProcessChains、mockElements
- [x] 清理不再使用的导入和引用
- [x] 确认 build 通过

## T8: 数据迁移与兼容
- [x] 已有 localStorage 数据（含旧 path）兼容处理
- [x] 自动扫描已有的 docs/ 目录验证端到端流程
- [x] 清除 localStorage 后重新建索引正常
