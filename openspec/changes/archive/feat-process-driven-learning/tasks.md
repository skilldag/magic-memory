# Tasks: 问题驱动推导学习流程

## T1: 数据类型定义

- [ ] 在 `types/index.ts` 中新增 `ConceptElement` 类型
- [ ] 在 `types/index.ts` 中新增 `ProcessStep` 类型（含 id/label/description/question/hint/leads_to_type/leads_to_id/is_core）
- [ ] 在 `types/index.ts` 中新增 `ProcessChain` 类型（含 id/name/steps）
- [ ] 扩展 `Concept` 类型：增加 `process?: { chain_id: string; step_index: number; role: string }` 和 `elements?: ConceptElement[]`
- [ ] 扩展 `ReviewRecord` 类型：增加 `process_state?: { user_flow: string[]; llm_flow: string[]; gaps: string[]; filled: boolean; compared: boolean }`

## T2: Mock 数据更新

- [ ] 为 Level 1 核心概念（0-VllmConfig, 1-Device, 2-Tensor, 5-GpuAllocator）补充 `process` 字段
- [ ] 定义一条"推理启动"过程链（含 5 个步骤，其中 2 个为可空缺步骤）
- [ ] 为上述 4 个概念各补充 2-4 个 `ConceptElement`
- [ ] 确保 `leads_to` 字段与过程链步骤顺序一致

## T3: ProcessCanvas 组件

- [ ] 新建 `src/components/ProcessCanvas.tsx`
- [ ] 接收 concept + processChain + 所有 concepts 作为 props
- [ ] 渲染过程骨架：步骤卡片横向排列，空缺步骤显示为虚线框 + 问号
- [ ] 拖拽交互：下方候选概念可拖入空缺
- [ ] 手动输入：用户也可在空缺处直接输入文字
- [ ] 提交按钮："我梳理完了，生成对照"
- [ ] 状态管理：用 `useState` 管理当前已填充的空缺列表
- [ ] LSP diagnostics 无报错

## T4: ComparisonPanel 组件

- [ ] 新建 `src/components/ComparisonPanel.tsx`
- [ ] 接收 user_flow + llm_flow 作为 props
- [ ] 并行渲染用户版本和 LLM 版本
- [ ] 差异项标记：匹配步骤 ✓、用户缺少步骤 ⚡、用户多余步骤 ⟳
- [ ] ⚡ 步骤可点击展开，展示说明文字和跳转按钮
- [ ] 跳转按钮："梳理这个概念"—调用 `onNavigate` 回调
- [ ] LSP diagnostics 无报错

## T5: LLM 对照生成（Mock 版本）

- [ ] 在 `src/utils/` 中新建 `processComparison.ts`
- [ ] 实现 `generateReferenceFlow(conceptId, allConcepts): ProcessStep[]` 函数
- [ ] 基于 `depends_on` / `leads_to` 链和 `process` 字段组装参考流程
- [ ] 实现 `diffFlows(userFlow, referenceFlow): DiffItem[]` 函数
- [ ] DiffItem 包含：步骤 id、匹配状态（match/missing/extra）、说明文本
- [ ] 单元测试：覆盖 VllmConfig 的流程对比（已知应漏掉 Tensor 和 ModelLoader）

## T6: ConceptDetailPanel 改造

- [ ] 删除现有的 `exploreTabs` 和 `learnTabs` 定义（第 65-77 行）
- [ ] 新增四个操作按钮：`['🔄 梳理过程', '⚖️ 对照验证', '🔗 探索关联', '📖 查阅文档']`
- [ ] 默认展示 ProcessCanvas（梳理过程面板）
- [ ] 第二个按钮展示 ComparisonPanel（需要先提交推导才可进入）
- [ ] "查阅文档"按钮展示 DocumentViewer（降级为辅助功能）
- [ ] 移除全局 viewMode 依赖，改为概念级状态驱动

## T7: KnowledgeGraphView 点击行为变更

- [ ] 删除第 156-161 行的全局 explore/review 模式切换按钮
- [ ] 保留 "探索" 和 "学习" 按钮但改为右侧面板视角控制（而非全局模式）
- [ ] 或者：完全移除模式切换，所有逻辑迁移到右侧面板内部
- [ ] 确保 hover 三个按钮（AI/?/手动）不受影响
- [ ] 确保图谱节点视觉编码不变

## T8: 节点状态视觉编码

- [ ] 在 `KnowledgeGraph.tsx` 中读取每个概念的 `process_state`
- [ ] 根据状态调整节点渲染样式：
  - 未点击 → 灰色填充 + 默认边框
  - 梳理中 → 蓝色边框脉冲动画
  - 比对完成有差异 → 琥珀色发光边框
  - 比对完成无差异 → 绿色稳定光
- [ ] 不影响现有布局和交互

## T9: Store 扩展

- [ ] 在 `knowledgeGraphStore.ts` 中新增 `updateProcessState(conceptId, state)` action
- [ ] 扩展 `startReview` 逻辑：提交推导后自动触发 review record 更新
- [ ] persist 配置同步更新（确保 process_state 被持久化）
- [ ] 新增 `getProcessChain(chainId): ProcessChain` selector

## T10: 集成测试

- [ ] e2e 测试：点击节点 → 看到过程骨架
- [ ] e2e 测试：拖拽或填写空缺 → 提交 → 看到对照结果
- [ ] e2e 测试：点击差异项 → 跳转到新概念
- [ ] e2e 测试：返回图谱 → 节点状态已更新
- [ ] 验证无 regression：hover 按钮仍正常工作
