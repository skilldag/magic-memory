# Magic Memory Web Annotator - 安装和启用指南

## 📋 前置要求

在开始之前，请确保你的系统已安装以下软件：

### 必需软件
- **Node.js**: 版本 18 或更高
- **npm**: 随 Node.js 一起安装
- **Bun**: JavaScript 运行时（用于后端服务）

### 可选软件
- **Git**: 用于版本控制

## 🚀 快速安装和启用

### 方式一：使用启动脚本（推荐）

#### 1. 进入项目目录
```bash
cd ~/source/magic-memory/magic-memory
```

#### 2. 运行启动脚本
```bash
./start.sh
```

这个脚本会自动：
- 检查并安装前端依赖
- 启动后端服务（端口 3001）
- 启动前端开发服务器（端口 3000）

#### 3. 访问应用
- 前端: http://localhost:3000
- 后端: http://localhost:3001

### 方式二：手动安装和启动

#### 1. 安装前端依赖
```bash
cd ~/source/magic-memory/magic-memory
npm install
```

#### 2. 启动后端服务
```bash
bun run server.ts
```

#### 3. 启动前端服务
```bash
npm run dev
```

#### 4. 访问应用
- 前端: http://localhost:3000
- 后端: http://localhost:3001

## 🔧 详细安装步骤

### 步骤 1: 检查 Node.js 和 npm

```bash
node --version
npm --version
```

如果未安装，请访问 [Node.js 官网](https://nodejs.org/) 下载安装。

### 步骤 2: 安装 Bun

```bash
curl -fsSL https://bun.sh/install | bash
```

或者使用 npm：
```bash
npm install -g bun
```

验证安装：
```bash
bun --version
```

### 步骤 3: 进入项目目录

```bash
cd ~/source/magic-memory/magic-memory
```

### 步骤 4: 安装前端依赖

```bash
npm install
```

这个过程可能需要几分钟，取决于你的网络速度。

### 步骤 5: 启动服务

#### 启动后端服务
```bash
bun run server.ts
```

你会看到类似这样的输出：
```
Server running on http://localhost:3001
Loaded 100 documents
```

#### 启动前端服务（新开一个终端窗口）
```bash
cd ~/source/magic-memory/magic-memory
npm run dev
```

你会看到类似这样的输出：
```
  VITE v6.2.0  ready in 1234 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

### 步骤 6: 访问应用

在浏览器中打开：
- http://localhost:3000

## 📱 使用指南

### 浏览文档
1. 从左侧侧边栏选择文档
2. 使用搜索框过滤文档
3. 按级别或分类筛选文档

### 添加注释
1. 在文档中选择文本
2. 点击工具栏中的注释类型按钮
3. 在右侧注释面板中查看和管理注释

### 导出文档
1. 点击工具栏中的导出按钮
2. 选择导出格式（JSON/Markdown/HTML）
3. 选择是否包含注释和元数据
4. 点击导出按钮下载文件

### 导入文档
1. 点击侧边栏中的导入按钮
2. 选择要导入的文件
3. 选择导入格式
4. 点击导入按钮

## 🛠️ 常见问题解决

### 问题 1: npm install 失败

**解决方案**：
```bash
rm -rf node_modules package-lock.json
npm install
```

### 问题 2: 端口被占用

**解决方案**：
修改 `vite.config.ts` 中的端口配置：
```typescript
export default defineConfig({
  server: {
    port: 3001,  // 改为其他端口
    // ...
  },
})
```

修改 `server.ts` 中的端口配置：
```typescript
const PORT = 3001  // 改为其他端口
```

### 问题 3: 文档加载失败

**解决方案**：
检查 `../docs` 目录是否存在且包含 Markdown 文件：
```bash
ls -la ../docs
```

### 问题 4: Bun 未安装

**解决方案**：
```bash
curl -fsSL https://bun.sh/install | bash
```

### 问题 5: 权限错误

**解决方案**：
```bash
chmod +x start.sh
```

## 🎯 开发模式 vs 生产模式

### 开发模式
```bash
npm run dev
```
- 支持热更新
- 显示详细错误信息
- 便于调试

### 生产模式
```bash
npm run build
npm run preview
```
- 优化后的代码
- 更快的加载速度
- 适合部署

## 📦 部署到生产环境

### 方式一：使用 Vercel

1. 安装 Vercel CLI：
```bash
npm install -g vercel
```

2. 部署：
```bash
vercel
```

### 方式二：使用 Docker

1. 创建 Dockerfile：
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

2. 构建镜像：
```bash
docker build -t magic-memory-annotator .
```

3. 运行容器：
```bash
docker run -p 3000:3000 magic-memory-annotator
```

### 方式三：使用传统服务器

1. 构建项目：
```bash
npm run build
```

2. 将 `dist` 目录上传到服务器
3. 配置 Nginx 或 Apache 指向 `dist` 目录

## 🔧 配置选项

### 环境变量

创建 `.env` 文件：
```env
VITE_API_URL=http://localhost:3001
```

### 修改端口

修改 `vite.config.ts`：
```typescript
export default defineConfig({
  server: {
    port: 3001,  // 修改前端端口
    // ...
  },
})
```

修改 `server.ts`：
```typescript
const PORT = 3001  // 修改后端端口
```

## 📊 系统要求

### 最低配置
- **CPU**: 双核处理器
- **内存**: 4GB RAM
- **磁盘**: 1GB 可用空间

### 推荐配置
- **CPU**: 四核处理器
- **内存**: 8GB RAM
- **磁盘**: 5GB 可用空间

## 🎓 学习资源

- [React 官方文档](https://react.dev/)
- [Vite 官方文档](https://vitejs.dev/)
- [Tailwind CSS 官方文档](https://tailwindcss.com/)
- [Bun 官方文档](https://bun.sh/)

## 🆘 获取帮助

如果遇到问题：

1. 查看本文档的"常见问题解决"部分
2. 检查浏览器控制台的错误信息
3. 查看终端的输出日志
4. 提交 Issue 到项目仓库

## 🎉 开始使用

安装完成后，你就可以：

1. 浏览和查看知识文档
2. 添加和管理注释
3. 导出和导入文档
4. 搜索和筛选文档
5. 享受现代化的用户界面

祝你使用愉快！
