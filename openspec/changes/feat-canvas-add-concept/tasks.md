# Tasks: feat-canvas-add-concept

## T1: 分离 KnowledgeGraph 增量更新的 effect 依赖
- [ ] 将 `useEffect([concepts, edges])` 改为 `useEffect([structuralKey])`
- [ ] 用 `useMemo` 基于 概念 ID 集 + 边集生成结构签名
- [ ] 验证：只改 content 不触发布局；增删节点/边正常触发

## T2: 实现 AddConceptDialog 组件
- [ ] 新建轻量 modal 组件，仅包含概念名输入
- [ ] Enter 确认 / 点击确认按钮提交
- [ ] 点击弹窗外区域取消
- [ ] CSS 与现有 ManualAddDialog/BatchLinkDialog 风格一致

## T3: 接入双击空白区添加概念
- [ ] KnowledgeGraphView 的 onBackgroundDoubleTap 中弹出 AddConceptDialog
- [ ] 确认后调用 store.addConcept() 创建概念
- [ ] 创建后自动选中新概念（selectConcept）
- [ ] 右侧面板展示空文档编辑区
- [ ] 验证：节点出现在图上 + 增量布局正常

## T4: ConceptDetailPanel 新增"更新关系和图谱"按钮
- [ ] 在文档编辑 Tab 底部新增 🔄 更新关系和图谱 按钮
- [ ] 点击按钮执行 reparseRelations 逻辑：
  - [ ] parseFrontmatter(当前文档 content)
  - [ ] matchTitlesToIds 解析 depends_on/leads_to/related
  - [ ] 更新 store 中该概念的引用数组
  - [ ] 对比边集 diff → 增删边
- [ ] 按钮 disabled 条件：content 为空或无 frontmatter
- [ ] 成功后显示轻量 toast/提示

## T5: 验证端到端流程
- [ ] 双击空白区添加概念 → 节点出现
- [ ] 双击已有概念节点 → 仍进入过程画板（不冲突）
- [ ] 编辑概念文档 frontmatter → 点击"更新关系" → 边变化
- [ ] 纯内容编辑（不改 frontmatter）→ 不触发图布局
- [ ] `updateConceptContent` 不触发增量布局 effect
