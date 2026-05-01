# Magic Memory - 设计文档

## 🎯 设计目标

创建一个基于 explore-drive-knowledge 理念的知识文档标注和查看系统，参考 plannotator 的架构设计，提供统一的 web 界面来查看和进行 annotator。

## 🏗️ 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Sidebar    │  │  Document    │  │ Annotation   │  │
│  │   (文档列表)  │  │   Viewer    │  │   Panel      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   React App      │
                    │   (前端应用)      │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Zustand Store  │
                    │   (状态管理)      │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   API Layer      │
                    │   (API 层)       │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Bun Server     │
                    │   (后端服务)      │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   File System    │
                    │   (文件系统)      │
                    └───────────────────┘
```

### 技术栈选择

#### 前端技术栈
- **React 19**: 最新的 React 版本，提供更好的性能和开发体验
- **TypeScript**: 类型安全，提高代码质量
- **Vite**: 快速的构建工具
- **Tailwind CSS 4**: 现代化的 CSS 框架
- **Zustand**: 轻量级状态管理
- **marked**: Markdown 解析
- **DOMPurify**: HTML 清理，防止 XSS 攻击

#### 后端技术栈
- **Bun**: 高性能的 JavaScript 运行时
- **TypeScript**: 类型安全
- **File System API**: 读取本地文档

### 核心组件设计

#### 1. DocumentViewer（文档查看器）
**职责**：
- 渲染 Markdown 内容
- 处理文本选择
- 显示注释高亮

**关键功能**：
```typescript
- renderMarkdown(): 渲染 Markdown 为 HTML
- handleTextSelection(): 处理文本选择事件
- handleAddAnnotation(): 添加注释
- handleAnnotationClick(): 处理注释点击
```

#### 2. AnnotationPanel（注释面板）
**职责**：
- 显示文档的所有注释
- 管理注释状态
- 支持回复功能

**关键功能**：
```typescript
- getStats(): 获取注释统计
- handleStatusChange(): 更改注释状态
- handleReplySubmit(): 提交回复
```

#### 3. Sidebar（侧边栏）
**职责**：
- 显示文档列表
- 提供搜索和筛选功能
- 支持文档导航

**关键功能**：
```typescript
- searchDocuments(): 搜索文档
- filterByLevel(): 按级别筛选
- filterByCategory(): 按分类筛选
```

#### 4. Toolbar（工具栏）
**职责**：
- 提供全局操作按钮
- 切换面板显示
- 显示应用状态

**关键功能**：
```typescript
- onSidebarToggle(): 切换侧边栏
- onAnnotationPanelToggle(): 切换注释面板
```

### 状态管理设计

#### DocumentStore（文档状态）
```typescript
interface DocumentStore {
  documents: Document[]           // 所有文档
  selectedDocument: Document | null  // 当前选中的文档
  isLoading: boolean              // 加载状态
  error: string | null           // 错误信息

  loadDocuments(): Promise<void>  // 加载文档
  selectDocument(doc): void       // 选择文档
  updateDocument(id, updates): void  // 更新文档
  searchDocuments(query): Document[]  // 搜索文档
  filterByLevel(level): Document[]   // 按级别筛选
  filterByCategory(category): Document[]  // 按分类筛选
}
```

#### AnnotationStore（注释状态）
```typescript
interface AnnotationStore {
  annotations: Annotation[]       // 所有注释
  selectedAnnotation: Annotation | null  // 当前选中的注释
  isLoading: boolean              // 加载状态
  error: string | null           // 错误信息

  loadAnnotations(documentId): Promise<void>  // 加载注释
  addAnnotation(annotation): void  // 添加注释
  updateAnnotation(id, updates): void  // 更新注释
  deleteAnnotation(id): void      // 删除注释
  selectAnnotation(annotation): void  // 选择注释
  getAnnotationsByDocument(documentId): Annotation[]  // 获取文档注释
  getStats(documentId): AnnotationStats  // 获取统计
  addReply(annotationId, reply): void  // 添加回复
}
```

### 数据流设计

#### 文档加载流程
```
用户操作 → Sidebar.onDocumentSelect()
         ↓
    DocumentStore.selectDocument()
         ↓
    DocumentViewer.render()
         ↓
    API 调用 /api/documents/:id
         ↓
    后端返回文档内容
         ↓
    marked.parse() → DOMPurify.sanitize()
         ↓
    React 渲染 HTML
```

#### 注释添加流程
```
用户选择文本 → DocumentViewer.handleTextSelection()
         ↓
    显示工具栏按钮
         ↓
    用户点击按钮 → DocumentViewer.handleAddAnnotation()
         ↓
    AnnotationStore.addAnnotation()
         ↓
    AnnotationPanel.render()
         ↓
    API 调用 POST /api/documents/:id/annotations
         ↓
    后端保存注释
         ↓
    返回成功响应
```

### API 设计

#### RESTful API 设计

**文档相关接口**：
```
GET    /api/documents              # 获取文档列表
GET    /api/documents/:id          # 获取文档详情
```

**注释相关接口**：
```
GET    /api/documents/:id/annotations  # 获取文档注释
POST   /api/documents/:id/annotations  # 添加注释
PUT    /api/annotations/:id            # 更新注释
DELETE /api/annotations/:id            # 删除注释
```

#### 响应格式

**成功响应**：
```json
{
  "success": true,
  "data": { ... }
}
```

**错误响应**：
```json
{
  "success": false,
  "error": "错误信息"
}
```

### 界面设计

#### 布局设计
```
┌─────────────────────────────────────────────────────────┐
│  Toolbar (工具栏)                                        │
├──────────┬──────────────────────────────────┬───────────┤
│          │                                  │           │
│ Sidebar │      Document Viewer             │ Annotation │
│          │                                  │  Panel    │
│          │                                  │           │
│          │                                  │           │
└──────────┴──────────────────────────────────┴───────────┘
```

#### 响应式设计
- **桌面端**（> 1024px）：三栏布局
- **平板端**（768px - 1024px）：两栏布局
- **移动端**（< 768px）：单栏布局

#### 颜色方案
```css
--color-primary: #3b82f6;      /* 主色 - 蓝色 */
--color-secondary: #8b5cf6;    /* 次色 - 紫色 */
--color-success: #10b981;      /* 成功 - 绿色 */
--color-warning: #f59e0b;      /* 警告 - 橙色 */
--color-error: #ef4444;        /* 错误 - 红色 */
--color-background: #ffffff;   /* 背景 - 白色 */
--color-surface: #f8fafc;      /* 表面 - 浅灰 */
--color-text: #1e293b;         /* 文本 - 深灰 */
--color-text-secondary: #64748b;  /* 次要文本 - 中灰 */
--color-border: #e2e8f0;       /* 边框 - 浅灰 */
```

### 性能优化

#### 前端优化
1. **代码分割**：使用 React.lazy() 和 Suspense
2. **虚拟滚动**：使用 OverlayScrollbars
3. **状态持久化**：使用 Zustand persist 中间件
4. **防抖节流**：对搜索和滚动事件进行优化

#### 后端优化
1. **缓存**：缓存文档内容
2. **流式响应**：支持大文档的流式传输
3. **并发处理**：使用 Bun 的并发特性

### 安全考虑

1. **XSS 防护**：使用 DOMPurify 清理 HTML
2. **CSRF 防护**：使用 CSRF Token
3. **输入验证**：验证所有用户输入
4. **权限控制**：实现基于角色的访问控制

### 可扩展性设计

#### 插件系统
```typescript
interface Plugin {
  name: string
  version: string
  init(): void
  destroy(): void
}
```

#### 主题系统
```typescript
interface Theme {
  colors: Record<string, string>
  fonts: Record<string, string>
  spacing: Record<string, number>
}
```

#### 国际化
```typescript
interface I18n {
  locale: string
  messages: Record<string, Record<string, string>>
  t(key: string): string
}
```

## 🚀 实现计划

### Phase 1: 基础功能 ✅
- [x] 项目结构搭建
- [x] 核心组件实现
- [x] 状态管理
- [x] 基础 API

### Phase 2: 核心功能 ✅
- [x] 文档浏览
- [x] 注释系统
- [x] 搜索筛选
- [x] 响应式设计

### Phase 3: 高级功能 🚧
- [ ] 导出导入
- [ ] 版本历史
- [ ] 协作功能
- [ ] 用户认证

### Phase 4: 优化完善 📋
- [ ] 性能优化
- [ ] 测试覆盖
- [ ] 文档完善
- [ ] 部署上线

## 📝 总结

这个 Web Annotator 系统基于 explore-drive-knowledge 理念，参考了 plannotator 的成熟架构，提供了一个现代化的知识文档标注和查看平台。

**核心优势**：
1. **现代化技术栈**：React 19 + TypeScript + Vite
2. **优秀的用户体验**：响应式设计 + 流畅的交互
3. **强大的功能**：文档浏览 + 注释系统 + 搜索筛选
4. **可扩展架构**：插件系统 + 主题系统 + 国际化

**适用场景**：
- 知识文档的阅读和标注
- 技术文档的协作编辑
- 学习笔记的整理和分享
- 团队知识库的建设

通过这个系统，用户可以方便地浏览知识文档，添加注释和反馈，促进知识的传播和交流。
