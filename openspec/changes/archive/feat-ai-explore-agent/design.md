# 设计：AI 概念探索

## 架构

```
┌────────────────────┐     POST /api/explore      ┌─────────────────────────────┐
│   ExploreDialog    │  ──────────────────────→   │   Local Server (Bun)        │
│   (React/Vite)     │                            │   @opencode-ai/plugin SDK   │
│                    │  ←──── 生成结果 JSON ────  │                            │
│   addConcept()     │                            │   ctx.client.llm.chat()     │
│   addEdge()        │                            │   或 HTTP 调 OpenCode      │
└────────────────────┘                            └──────────┬──────────────────┘
                                                             │
                                                             │ 无状态请求
                                                             ▼
                                                  ┌─────────────────────┐
                                                  │  OpenCode LLM       │
                                                  │  (当前配置的provider)│
                                                  └─────────────────────┘
```

## 数据流

1. 用户填问题 → 点击"添加到图谱并探索"
2. 前端 POST `{ sourceConcept: { id, title, problem }, userQuestion: string }` 到 server
3. Server 构造 system prompt + user prompt，调 OpenCode LLM（无 session 上下文）
4. LLM 返回结构化 JSON：
```json
{
  "title": "概念名称",
  "problem": "它所解决的问题",
  "gap_anticipate": "可能产生的疑问",
  "content": "# 概念名称\n\nMarkdown 内容...",
  "relationType": "leads_to"
}
```
5. Server 返回结果给前端
6. 前端 `addConcept()` + `addEdge()` 加入 store
7. 图谱自动刷新

## 关于「无上下文」

调 OpenCode 的 LLM 时，只传 `system + user message`，不传任何历史对话消息：

```
system: "你是一个知识图谱概念生成助手。基于用户提供的源概念和探索问题，生成一个新的概念节点。输出 JSON。"
user: "源概念: PagedAttention - 分页注意力\n源问题: 如何高效管理长序列的 KV Cache？\n探索问题: 都需要支持哪些不同的 LLM Provider？"
```

## Server 实现

用 Bun 起一个轻量 server，注册 `/api/explore` 端点。通过 `@opencode-ai/plugin` SDK 的 LLM 调用能力（非 `session.prompt`）实现无状态生成。

如果 OpenCode SDK 不支持直接 LLM completion，则 fallback 到 HTTP 调 OpenCode 的本地 API。
