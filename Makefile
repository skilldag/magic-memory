.PHONY: run dev server explore clean restart stop

WD = .

# 启动所有服务（前端 + 后端 + AI 探索，后台进程需手动 kill）
run:
	@echo "🚀 启动所有服务..."
	@echo "📦 检查依赖..."
	@if [ ! -d "$(WD)/node_modules" ]; then cd $(WD) && npm install; fi
	@cd $(WD) && bun run server/explore.ts &
	@sleep 2 && \
	echo "✅ 前端: http://localhost:3000" && \
	echo "✅ 后端/API: http://localhost:4321" && \
	echo "按 Ctrl+C 停止前端（后台进程需手动 kill）" && \
	cd $(WD) && npm run dev

# 停止所有服务
stop:
	@echo "🛑 停止所有服务..."
	@lsof -ti:3000 -ti:4321 2>/dev/null | xargs kill -9 2>/dev/null; true

# 重启所有服务
restart: stop
	@echo "🔄 重启所有服务..."
	@echo "📦 检查依赖..."
	@if [ ! -d "$(WD)/node_modules" ]; then cd $(WD) && npm install; fi
	@cd $(WD) && bun run server/explore.ts &
	@sleep 2 && \
	echo "✅ 前端: http://localhost:3000" && \
	echo "✅ 后端/API: http://localhost:4321" && \
	echo "按 Ctrl+C 停止前端（后台进程需手动 kill）" && \
	cd $(WD) && npm run dev

# 停止所有服务
stop:
	@echo "🛑 停止所有服务..."
	@lsof -ti:3000 -ti:3001 -ti:4321 2>/dev/null | xargs kill -9 2>/dev/null; true

# 重启所有服务
restart: stop
	@echo "🔄 重启所有服务..."
	@if [ ! -d "$(WD)/node_modules" ]; then cd $(WD) && npm install; fi
	@cd $(WD) && bun run server.ts &
	@cd $(WD) && bun run server/explore.ts &
	@sleep 2 && \
	echo "✅ 前端: http://localhost:3000" && \
	echo "✅ 后端: http://localhost:3001" && \
	echo "✅ AI:   http://localhost:4321" && \
	echo "按 Ctrl+C 停止前端（后台进程需手动 kill）" && \
	cd $(WD) && npm run dev

# 只启动前端开发服务器
dev:
	@echo "🔧 启动前端 (port 3000)..."
	@cd $(WD) && npm run dev

# 只启动后端 API 服务器（含 AI 探索）
server:
	@echo "🔌 启动后端 API (port 4321)..."
	@cd $(WD) && bun run server/explore.ts

# 清理构建产物
clean:
	rm -rf $(WD)/dist/
	rm -rf $(WD)/.vite/
	@echo "✅ 已清理"
