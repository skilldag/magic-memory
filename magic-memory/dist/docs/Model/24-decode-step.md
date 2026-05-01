# 24 - 闹钟 → Decode Step

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 24 | ⏰ 闹钟 | Decode Step |

**记忆口诀**: 解码如闹钟每步滴答产生 token

---

## vLLM 概念

**Decode Step** 是解码步骤：

```
输入: [Hello]
  ↓ Forward
logits → Sampler → new_token
  ↓
输出: [Hello, world]
```

**记忆故事**:
- 闹钟滴答 = token 滴答生成

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Sampler](./20-cigarette.md) | 20 | 香烟 |

---

*闹钟滴答，token 生成。*