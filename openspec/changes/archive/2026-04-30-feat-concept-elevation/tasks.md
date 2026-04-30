# Tasks: 概念提升功能

## T1: 修改 DocumentViewer 组件 - 添加概念提升按钮

- [x] 在 DocumentViewer.tsx 中找到注释工具栏区域
- [x] 将"建议修改"按钮改为"概念提升"按钮
- [x] 修改按钮文字： "建议修改" → "概念提升"

## T2: 实现概念提升逻辑 - 创建概念并建立关系

- [x] 在 DocumentViewer.tsx 或新建 hook 实现概念提升逻辑
- [x] 获取当前选中的概念（从 knowledgeGraphStore 的 selectedConcept）
- [x] 使用选中文字作为新概念的 title
- [x] 调用 createConceptWithEdges 创建概念，relationType 为 'leads_to'
- [x] 新概念 path 置空（或设置为 './docs/user/pending.md'）

## T3: 实现跳转和聚焦功能

- [x] 创建概念后，更新 knowledgeGraphStore 的 selectedConcept 为新概念
- [x] 触发视图切换：从文档视图切换到知识图谱视图（viewMode = 'knowledge-graph'）
- [x] 确保知识图谱组件能够自动聚焦到 selectedConcept

## T4: 边界情况处理

- [x] 处理未选中概念的情况：提示用户先在知识图谱中选择一个概念
- [x] 处理未选中文字的情况：提示用户先选中要提升的文字
- [x] 处理选中文字为空或只有空白字符的情况

## T5: 测试验证

- [ ] 在知识图谱视图中选择任意概念节点
- [ ] 在文档中选中一段文字
- [ ] 点击"概念提升"按钮
- [ ] 验证新概念被创建并显示在知识图谱中
- [ ] 验证从原概念到新概念有 leads_to 边
- [ ] 验证视图自动切换到知识图谱并聚焦到新概念