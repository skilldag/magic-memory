.PHONY: run dev server explore clean

# 启动所有服务（前端 + 后端 + AI 探索，后台进程需手动 kill）
run:
	@echo "🚀 启动所有服务..."
	@echo "📦 检查依赖..."
	@if [ ! -d "node_modules" ]; then npm install; fi
	@bun run server.ts &
	@bun run server/explore.ts &
	@sleep 2 && \
	echo "✅ 前端: http://localhost:3000" && \
	echo "✅ 后端: http://localhost:3001" && \
	echo "✅ AI:   http://localhost:4321" && \
	echo "按 Ctrl+C 停止前端（后台进程需手动 kill）" && \
	npm run dev

# 只启动前端开发服务器
dev:
	@echo "🔧 启动前端 (port 3000)..."
	@npm run dev

# 只启动后端 API 服务器
server:
	@echo "🔌 启动后端 API (port 3001)..."
	@bun run server.ts

# 只启动 AI 探索服务
explore:
	@echo "🤖 启动 AI 探索 (port 4321)..."
	@bun run server/explore.ts

# 清理构建产物
clean:
	rm -rf dist/
	rm -rf .vite/
	@echo "✅ 已清理"
