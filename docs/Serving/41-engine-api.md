# 41 - 蜥蜴 → Engine API

> [← 返回目录](../level-3/README.md)

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 41 | 🦎 蜥蜴 | Engine API |

**记忆口诀**: API 如蜥蜴爬行接口层

---

## vLLM 概念

**Engine API** 引擎接口:

```rust
pub trait LlmEngine {
    fn add_request(&mut self, request: Request) -> RequestId;
    fn step(&mut self) -> Vec<RequestOutput>;
    fn abort(&mut self, request_id: RequestId);
}
```

---

*蜥蜴爬行，API 接口。*