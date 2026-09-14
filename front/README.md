# AI Customer Service Agent Frontend

React / Vite 客服网页，连接现有 FastAPI 接口；会话状态、流式回复和本地记录由 assistant-ui 与现有适配器管理。

## 界面组件

- Ant Design：按钮、表单及校验、弹窗、抽屉、标签页、提示、上传和图片预览。
- Ant Design X：会话列表、欢迎区、推荐问题、消息气泡、输入框和附件。
- 浅深主题统一由 XProvider 配置；保留本地主题偏好，尊重系统减少动画设置。
- 本地打包 Noto Sans SC 可变字体，按字符范围加载，不依赖外部字体 CDN。
- `src/index.css` 仅维护网页布局、响应式尺寸与排版；通用控件的外观和交互由组件库提供。
- 管理页面按需加载，保留组件库的弹窗退出动画。

## 开发与构建

```powershell
npm install
npm run dev
npm run build
```

默认 API 地址为 `http://127.0.0.1:8000`，可通过 `VITE_API_BASE_URL` 配置。

## 输入框回归检查

运行 `npm test`，检查中文输入法组词及选词、连续字母与光标位置、换行和单次发送。
测试在输入事件结束时直接检查文本，防止异步状态刷新掩盖旧值回写。
输入框使用 assistant-ui 提供的 `flushTapSync` 同步更新受控值，与其内置输入组件保持一致。

## 隔离界面验收

以下服务只读写进程内的测试数据，不访问真实模型或企业知识库。分别在两个终端运行：

```powershell
node scripts/ui-fixture.mjs
```

```powershell
$env:VITE_API_BASE_URL = 'http://127.0.0.1:8011'
npm run dev -- --port 5174 --strictPort
```

打开 `http://127.0.0.1:5174`。可检查推荐问题与流式回复、输入“慢速测试”后停止生成、附件预览及纯图片发送、表单保存和校验、知识库上传及删除确认。
`scripts/fixtures` 提供测试图片和文档；示例失败文档会返回 HTTP 500，用于检查页面内错误提示。
测试请求摘要见 `http://127.0.0.1:8011/__requests`，关闭服务即清除数据。

界面验收覆盖浅深主题、桌面与窄屏布局。隔离验收不代表真实模型、鉴权或知识库索引服务已通过验收。
