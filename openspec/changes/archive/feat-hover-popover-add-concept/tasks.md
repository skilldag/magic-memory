# Tasks: 图谱节点悬停浮层

## T1: 创建 ConceptHoverPopover 组件
- [ ] 在 KnowledgeGraphView.tsx 中内联新增 ConceptHoverPopover 组件
- [ ] 接收 concept / x / y / containerWidth / containerHeight / onExplore / onManualAdd / onClose 参数
- [ ] 实现位置计算逻辑（默认右侧 +8px，超出边界自动切换到左侧）
- [ ] 实现浮层样式和两个按钮
- [ ] 实现鼠标 enter/leave 事件与 scheduleHideHoverActions 的集成

## T2: 替换现有固定操作栏
- [ ] 删除 KnowledgeGraphView.tsx 中 L351-398 的固定操作栏（absolute top-3 right-16）
- [ ] 在 onHoverConcept handler 中记录 hoverConcept 的同时记录 (x, y, containerWidth, containerHeight)
- [ ] 在 hoverConcept 非空时渲染 ConceptHoverPopover 替代固定操作栏
- [ ] 确保 actionConcept 和所有 dialog handler 正确绑定到浮层按钮

## T3: 异常状态处理
- [ ] 确保浮层不超出图容器边界
- [ ] 确保浮层出现时不影响图的交互（pointer-events-auto 只影响浮层自身）
- [ ] 确保鼠标快速移动经过多个节点时不会出现残影

## T4: 验证
- [ ] LSP diagnostics 无报错
- [ ] 构建通过（npm run build）
- [ ] 手动验证：探索模式选中概念 → 悬停节点 → 浮层出现 → 点击 AI 生成 → dialog 打开 → 确认关联
