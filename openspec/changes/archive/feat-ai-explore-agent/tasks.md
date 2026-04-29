# Tasks: AI 概念探索

## T1: 创建本地生成 server
- [ ] 在 `magic-memory/` 下创建 `server/explore.ts`（Bun server）
- [ ] 注册 `POST /api/explore` 端点
- [ ] 通过 `@opencode-ai/plugin` SDK 或 HTTP 调用 OpenCode LLM
- [ ] 构造 system prompt（输出固定 JSON 结构）
- [ ] 返回结构化概念数据给前端

## T2: 前端接入
- [ ] ExploreDialog 提交时调 `POST /api/explore`
- [ ] 等待生成结果（loading 状态）
- [ ] 用返回结果调 `addConcept()` + `addEdge()`
- [ ] 自动关闭 dialog 并选中新概念

## T3: 验证集成
- [ ] 构建验证
- [ ] 端到端测试：输入问题 → 生成概念 → 图谱刷新
