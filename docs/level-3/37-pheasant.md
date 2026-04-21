# 37 - 山鸡 → Decode

> [← 返回目录](../level-3/README.md)

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 37 | 🦃 山鸡 | Decode |

**记忆口诀**: Decode 如野鸡慢走解码

---

## vLLM 概念

**Decode** 解码阶段:

```
前一个 token
  ↓ [逐个前向]
新 token
  ↓
KV Cache 追加
```

**特点**:
- 逐 token 生成
- 内存密集
- 依赖 KV Cache

---

## 关联概念

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [Prefill](./36-deer.md) | 36 | 山鹿 |

---

*鸡慢走，decode 逐 token。*