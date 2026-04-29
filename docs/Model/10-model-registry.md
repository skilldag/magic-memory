# 10 - 棒球 → ModelRegistry

> **[← 返回目录](../level-2/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 10 | ⚾ 棒球 | ModelRegistry |

**记忆口诀**: 模型注册如棒球规则有限制（10个字符内命名）

---

## vLLM 概念

**ModelRegistry** 是模型注册中心：

```rust
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// 注册模型
    pub fn register(&mut self, name: &str, info: ModelInfo);

    /// 获取模型
    pub fn get(&self, name: &str) -> Option<&ModelInfo>;
}
```

**记忆故事**:
- 棒球规则限制 = 模型名 10 字符限制

---

## 核心功能

| 功能 | 说明 | 记忆 |
|------|------|------|
| **register** | 注册模型 | 添加 |
| **get** | 获取模型信息 | 查询 |
| **list** | 列出所有模型 | 列表 |

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [ModelLoader](./11-chopsticks.md) | 11 | 筷子 |
| [Model](./12-highchair.md) | 12 | 婴儿椅 |

---

*棒球有规则，模型注册有限制。*