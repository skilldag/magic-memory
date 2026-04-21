# 49 - 湿狗 → Cache Eviction

> [← 返回目录](../level-3/README.md)

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 49 | 🐕 湿狗 | Cache Eviction |

**记忆口诀**: 缓存驱逐如湿狗甩水

---

## vLLM 概念

**Cache Eviction** 缓存驱逐:

```
显存不足
  ↓ [LRU/FIFO 策略]
驱逐旧 Cache
  ↓
释放显存
```

---

*狗甩水，Cache 驱逐。*