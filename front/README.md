# AI Customer Service Agent Frontend

这是 AI Customer Service Agent 的演示客户端，界面基于 Vite、React 与 assistant-ui 构建。

当前阶段只完成前端界面和本地演示交互，**尚未请求 FastAPI**。本地演示回复集中在：

```text
src/lib/demo-chat-adapter.ts
```

下一阶段会在这个适配器中学习：

```text
输入框消息
  -> fetch 请求 FastAPI /chat
  -> 接收 ChatResponse
  -> 将 answer 交给 assistant-ui 显示
```

## 本地启动

```powershell
npm run dev
```

## 构建检查

```powershell
npm run build
```

## 第三方组件

界面使用 [assistant-ui](https://github.com/assistant-ui/assistant-ui)（MIT License）。
