# AI 知识图谱概念探索

## 问题

当前 ExploreDialog 提交问题后只是做了字符串拼接，没有真正的 AI 生成。

## 方案

前端提交问题 → 本地 server（Node/Bun）通过 `@opencode-ai/plugin` SDK 调 OpenCode 的 LLM → 返回结构化 JSON → 写入 store → 图谱刷新。

### 关键要求

- 请求不携带当前对话上下文（无状态调用）
- LLM 输出固定的 JSON 结构（概念名、问题、认知缺口、内容、关联关系）
- 前端等待生成完成后自动刷新图谱

### 为什么不直接调 LLM API

因为用户想用 OpenCode 来发起请求，复用当前配置的 provider/model。
