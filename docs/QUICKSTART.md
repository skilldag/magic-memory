# Magic Memory Web Annotator - 快速启动指南

## 🎯 项目概述

这是一个基于 explore-drive-knowledge 理念的知识文档标注和查看系统，参考了 plannotator 的架构设计。

## 🚀 快速启动

### 方式一：使用启动脚本（推荐）

```bash
cd magic-memory
make run
```

### 方式二：手动启动

**启动后端服务：**
```bash
cd magic-memory
bun run server.ts
```

**启动前端服务：**
```bash
cd magic-memory
npm run dev
```

## 📋 功能说明

### 1. 文档浏览
- 从左侧侧边栏选择文档
- 支持按级别、分类筛选
- 支持关键词搜索

### 2. 注释系统
- 选择文本后添加注释
- 支持四种注释类型：评论、问题、建议、纠正
- 支持回复和状态管理

### 3. 界面布局
- **左侧**: 文档列表和筛选
- **中间**: 文档内容查看
- **右侧**: 注释面板

## 🏗️ 架构设计

### 前端架构
```
React 19 + TypeScript + Vite
├── Components (UI 组件)
├── Store (Zustand 状态管理)
├── Utils (工具函数)
└── Hooks (自定义 Hooks)
```

### 后端架构
```
Bun + TypeScript
├── API 路由
├── 文档加载
└── 注释管理
```

## 📁 项目结构

```
magic-memory/
├── src/
│   ├── components/       # React 组件
│   │   ├── DocumentViewer.tsx    # 文档查看器
│   │   ├── AnnotationPanel.tsx   # 注释面板
│   │   ├── Sidebar.tsx           # 侧边栏
│   │   └── Toolbar.tsx           # 工具栏
│   ├── store/           # Zustand 状态管理
│   │   ├── documentStore.ts      # 文档状态
│   │   └── annotationStore.ts    # 注释状态
│   ├── types/           # TypeScript 类型
│   ├── App.tsx          # 主应用
│   └── main.tsx         # 入口文件
├── server.ts            # 后端服务
├── Makefile             # 启动脚本
└── README.md            # 项目文档
```

## 🔧 开发指南

### 添加新功能

1. **添加新组件**：在 `src/components/` 中创建
2. **添加新状态**：在 `src/store/` 中创建
3. **添加新类型**：在 `src/types/` 中定义

### 代码规范

- 使用 TypeScript 类型定义
- 遵循 React Hooks 规范
- 使用 Tailwind CSS 样式
- 避免不必要的注释

## 🎨 界面设计

### 颜色方案
- 主色：蓝色 (#3b82f6)
- 次色：紫色 (#8b5cf6)
- 成功：绿色 (#10b981)
- 警告：橙色 (#f59e0b)
- 错误：红色 (#ef4444)

### 响应式设计
- 桌面端：三栏布局
- 平板端：两栏布局
- 移动端：单栏布局

## 📊 数据流

```
用户操作 → 组件事件 → Store 更新 → UI 重新渲染
         ↓
    API 调用 → 后端处理 → 数据返回
```

## 🔌 API 接口

### 文档相关
- `GET /api/documents` - 获取文档列表
- `GET /api/documents/:id` - 获取文档详情

### 注释相关
- `GET /api/documents/:id/annotations` - 获取文档注释
- `POST /api/documents/:id/annotations` - 添加注释
- `PUT /api/annotations/:id` - 更新注释
- `DELETE /api/annotations/:id` - 删除注释

## 🚀 部署指南

### 构建生产版本

```bash
npm run build
```

### 预览生产版本

```bash
npm run preview
```

### 环境变量

创建 `.env` 文件：

```env
VITE_API_URL=http://localhost:3001
```

## 🐛 常见问题

### 1. 依赖安装失败
```bash
rm -rf node_modules package-lock.json
npm install
```

### 2. 端口被占用
修改 `vite.config.ts` 和 `server.ts` 中的端口配置

### 3. 文档加载失败
检查 `../docs` 目录是否存在且包含 Markdown 文件

## 📝 待办事项

- [ ] 实现导出和导入功能
- [ ] 集成现有文档系统
- [ ] 添加用户认证
- [ ] 实现版本历史
- [ ] 添加协作功能
- [ ] 优化性能

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

MIT License

## 📞 联系方式

如有问题或建议，请提交 Issue。
