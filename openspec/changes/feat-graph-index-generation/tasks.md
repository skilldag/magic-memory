# Tasks: feat-graph-index-generation

## T1: 添加空白图谱引导 UI
- [ ] KnowledgeGraphView 中检测 concepts 为空时，显示设置路径/选择模式界面
- [ ] 输入 docs 路径的 input + [自动扫描] [手动添加] 两个按钮
- [ ] localStorage 有数据时跳过引导，正常渲染图谱

## T2: 实现后端 /api/scan-docs 接口
- [ ] server.ts 新增 POST /api/scan-docs { path }
- [ ] 递归遍历目录下所有 .md 文件
- [ ] 解析 frontmatter → ConceptIndex
- [ ] 无 frontmatter 的文件调 LLM 分析内容→生成 frontmatter
- [ ] 构建 edges，返回 { concepts, edges }

## T3: 增强 conceptParser.ts
- [ ] parseFrontmatter 函数复用现有实现
- [ ] 添加 title-to-id 模糊匹配函数
- [ ] 添加从文件路径推断 category 的函数

## T4: 实现后端 /api/manual-add-concept 接口
- [ ] server.ts 新增 POST /api/manual-add-concept { title, context }
- [ ] LLM 分析返回结构化概念数据
- [ ] 标题匹配到已有概念 ID

## T5: 前端手动添加流程
- [ ] 增强现有的 / 按钮（QuickExploreDialog），输入概念名
- [ ] 调用 LLM → 接收结构化数据 → 匹配 → 创建索引
- [ ] 写入 .md 文件
- [ ] 更新 localStorage → 图谱刷新

## T6: 实现文档缺失 LLM 生成
- [ ] ConceptDetailPanel 中 docContent 为 null 时显示 LLM 按钮
- [ ] 调用 /api/generate-doc 生成全文
- [ ] 写入 {docs_path}/{path}
- [ ] 重新加载显示

## T7: 移除 mockGraphData.ts 依赖
- [ ] loadGraph() 不再从 mockGraphData 初始化
- [ ] 删除 mockConcepts、mockEdges、mockProcessChains、mockElements
- [ ] 清理不再使用的导入和引用
- [ ] 确认 build 通过

## T8: 数据迁移与兼容
- [ ] 已有 localStorage 数据（含旧 path）兼容处理
- [ ] 自动扫描已有的 docs/ 目录验证端到端流程
- [ ] 清除 localStorage 后重新建索引正常
