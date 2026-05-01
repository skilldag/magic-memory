# Magic Memory Web Annotator

基于 explore-drive-knowledge 的知识文档标注和查看系统。

## 功能特性

- 📚 **文档浏览**: 支持查看和浏览知识文档
- 💬 **注释系统**: 支持添加评论、问题、建议和纠正
- 🔍 **搜索过滤**: 按级别、分类和关键词搜索文档
- 🎨 **界面友好**: 现代化的 UI 设计，支持响应式布局
- 💾 **本地存储**: 使用浏览器本地存储保存注释和状态

## 技术栈

- **前端**: React 19 + TypeScript + Vite
- **样式**: Tailwind CSS 4
- **状态管理**: Zustand
- **Markdown**: marked + DOMPurify
- **滚动**: OverlayScrollbars

## 项目结构

```
magic-memory/
├── src/
│   ├── components/       # React 组件
│   │   ├── DocumentViewer.tsx
│   │   ├── AnnotationPanel.tsx
│   │   ├── Sidebar.tsx
│   │   └── Toolbar.tsx
│   ├── store/           # Zustand 状态管理
│   │   ├── documentStore.ts
│   │   └── annotationStore.ts
│   ├── utils/           # 工具函数
│   ├── hooks/           # 自定义 Hooks
│   ├── types/           # TypeScript 类型定义
│   ├── App.tsx          # 主应用组件
│   ├── main.tsx         # 应用入口
│   └── index.css        # 全局样式
├── public/              # 静态资源
├── index.html           # HTML 模板
├── package.json         # 项目配置
├── tsconfig.json        # TypeScript 配置
└── vite.config.ts       # Vite 配置
```

## 快速开始

### 安装依赖

```bash
cd magic-memory
npm install
```

### 启动开发服务器

```bash
npm run dev
```

访问 http://localhost:3000

### 构建生产版本

```bash
npm run build
```

### 预览生产版本

```bash
npm run preview
```

## 使用说明

### 浏览文档

1. 从左侧侧边栏选择文档
2. 使用搜索框过滤文档
3. 按级别或分类筛选文档

### 添加注释

1. 在文档中选择文本
2. 点击工具栏中的注释类型按钮
3. 在右侧注释面板中查看和管理注释

### 管理注释

1. 在注释面板中查看所有注释
2. 点击注释查看详情
3. 添加回复或更改状态
4. 删除不需要的注释

## API 接口

### 获取文档列表

```
GET /api/documents
```

### 获取文档详情

```
GET /api/documents/:id
```

### 获取文档注释

```
GET /api/documents/:id/annotations
```

### 添加注释

```
POST /api/documents/:id/annotations
```

### 更新注释

```
PUT /api/annotations/:id
```

### 删除注释

```
DELETE /api/annotations/:id
```

## 开发指南

### 添加新组件

1. 在 `src/components/` 中创建新组件
2. 使用 TypeScript 定义 Props 接口
3. 遵循现有的代码风格

### 添加新状态

1. 在 `src/store/` 中创建新的 store
2. 使用 Zustand 的 `create` 函数
3. 使用 `persist` 中间件持久化状态

### 样式指南

- 使用 Tailwind CSS 类名
- 遵循响应式设计原则
- 使用语义化的 HTML 标签

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue。
