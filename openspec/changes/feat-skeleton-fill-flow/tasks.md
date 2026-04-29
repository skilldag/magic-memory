# Tasks: 骨架填充流程

## T1: 扩展类型定义

- [ ] 在 `types/index.ts` 中新增 `BaseQuestion` 接口（id/conceptId/question/targetConceptId/hint/order）
- [ ] 在 `types/index.ts` 中新增 `UserQuestion` 接口（id/conceptId/question/context/status/convertedTo/createdAt）
- [ ] 在 `types/index.ts` 中新增 `CanvasHistoryItem` 接口（conceptId/view）
- [ ] 扩展 `Concept` 接口：增加 `hierarchy?: { parentId, level, order }` 和 `baseQuestions?: BaseQuestion[]`

## T2: Mock 数据补充

- [ ] 在 `mockGraphData.ts` 中新增 `mockBaseQuestions`（为 id 0/1/2 各添加 2-3 个 BaseQuestion）
- [ ] 更新 `mockConcepts` 中 id 0/1/2 的导出，加入 `baseQuestions` 字段

## T3: 骨架节点生成工具

- [ ] 在 `processComparison.ts` 中新增 `SkeletonNodeDef` 类型
- [ ] 实现 `generateSkeletonNodes(concept, chain, allConcepts): SkeletonNodeDef[]`
- [ ] 基于 process chain 的 steps 生成 gap/current 节点
- [ ] 无 chain 时降级使用 `generateGenericChain` 兜底

## T4: Store 扩展

- [ ] 在 `knowledgeGraphStore.ts` 中新增 `questions: UserQuestion[]`
- [ ] 新增 `canvasHistory: CanvasHistoryItem[]`
- [ ] 新增 `skeletonCompleted: Set<string>`
- [ ] 新增 actions：`addQuestion`, `setConceptPanelMode`, `markSkeletonCompleted`, `pushHistory`, `popHistory`

## T5: ProcessCanvas 骨架模式

- [ ] 新增 `skeletonMode`, `skeletonNodes`, `onSkeletonSubmit`, `onOpenQuestion` props
- [ ] 渲染骨架模式界面：空缺卡片（带引导问题）+ 进度条 + 候选概念区 + 提交按钮
- [ ] 实现拖拽填充交互（HTML5 drag-and-drop + dataTransfer）
- [ ] 实现自定义节点创建入口
- [ ] 实现提交验证逻辑（比对 correctConceptId）
- [ ] 验证结果展示（正确/错误逐项反馈）
- [ ] LSP diagnostics 无报错

## T6: KnowledgeGraphView 骨架集成

- [ ] 新增面包屑导航栏（图谱 > 概念名 [填充/画板]）
- [ ] 新增 `shouldShowSkeleton` 判断逻辑
- [ ] 将 `skeletonMode` 和 `skeletonNodes` 传给 ProcessCanvas
- [ ] 接入 `generateSkeletonNodes` 工具函数
- [ ] 新增提问对话框（QuestionDialog）
- [ ] LSP diagnostics 无报错

## T7: ConceptDetailPanel 问题集 tab

- [ ] 新增「问题集」tab（actions 数组扩展）
- [ ] 展示当前概念关联的 `UserQuestion` 列表
- [ ] 实现「转为新概念」按钮 → 调用 `addConcept` + `addEdge`
- [ ] 实现「转为流程步骤」按钮 → 补充到 `ProcessChain`
- [ ] 在问题集底部新增提问输入区
- [ ] LSP diagnostics 无报错

## T8: 集成验证

- [ ] `npx tsc --noEmit` 无类型错误
- [ ] `npm run dev` 正常启动
- [ ] 手动验证：双击概念 → 骨架画板显示空缺节点 + 引导问题
- [ ] 手动验证：拖拽候选概念到空缺 → 空缺被填充
- [ ] 手动验证：提交 → 正确/错误逐项反馈
- [ ] 手动验证：提问 → 出现在问题集 tab
- [ ] 手动验证：转化问题为新概念 → 图谱出现新节点
- [ ] 回归验证：已有 ProcessCanvas 自由画板不受影响
