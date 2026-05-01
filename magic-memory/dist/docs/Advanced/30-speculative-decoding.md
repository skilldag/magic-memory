# 30 - 三菱 → Speculative Decoding

> [← 返回目录](../level-3/README.md)

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 30 | 🔺 三菱 | Speculative Decoding |

**记忆口诀**: 推测解码如三菱标志三分支

---

## vLLM 概念

**Speculative Decoding** 推测解码:

```
Draft Model → Verifier → Target Model
  (快速生成)  (验证)     (最终确认)
```

### 核心技术

| 概念 | 说明 |
|------|------|
| Draft | 起草 token |
| Verify | 验证 |
| 接受率 | 通常 80-90% |

---

## 关联概念

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Draft Token](./31-yam.md) | 31 | 山药 |
| [Verifier](./32-fan.md) | 32 | 扇子 |

---

*三菱三分支，推测解码快。*