#!/bin/bash

echo "🚀 启动 Magic Memory"

echo "📦 检查依赖..."
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    npm install
fi

echo "🔧 启动后端服务..."
bun run server.ts &
SERVER_PID=$!

echo "🤖 启动 AI 探索服务..."
bun run server/explore.ts &
EXPLORE_PID=$!

echo "⏳ 等待服务启动..."
sleep 3

echo "🎨 启动前端开发服务器..."
npm run dev &
FRONTEND_PID=$!

echo "✅ 服务已启动!"
echo "📱 前端:   http://localhost:3000"
echo "🔌 后端:   http://localhost:3001"
echo "🤖 AI 探索: http://localhost:4321"
echo ""
echo "按 Ctrl+C 停止所有服务"

trap "kill $SERVER_PID $EXPLORE_PID $FRONTEND_PID; exit" INT TERM

wait
