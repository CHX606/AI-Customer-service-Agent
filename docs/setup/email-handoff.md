# 邮件转人工流程

用户填写事情并提交后，服务先将申请和最近40条文本会话保存在租户数据库，再通过SMTP发出通知。后台“人工待办”可查看详情，设置“待处理 / 处理中 / 已处理”及处理备注。备注供管理人员使用，不会自动发回用户聊天。

## 配置

在本地 `.env` 或服务器 `.env.production` 设置以下项目，然后重启后端。

```dotenv
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-account@gmail.com
SMTP_PASSWORD=
SMTP_FROM=your-account@gmail.com
SMTP_SECURITY=starttls
SMTP_TIMEOUT_SECONDS=15
HANDOFF_NOTIFICATION_TO=your-account@gmail.com
HANDOFF_ADMIN_URL=https://your-service.example/manage
```

Gmail需开启两步验证并创建[应用专用密码](https://support.google.com/accounts/answer/185833)，填写在SMTP_PASSWORD中，不使用普通登录密码。通知邮箱可以与发信邮箱相同。HANDOFF_ADMIN_URL可留空，此时邮件包含申请编号和问题描述。生产环境应填实际服务的后台地址。

HANDOFF_NOTIFICATION_TO只用于default租户。其他租户使用HANDOFF_NOTIFICATION_RECIPIENTS的JSON映射，如 `{"tenant_a":"operator@example.com"}`；缺少收件人时仍保存申请，不向其他租户邮箱投递。

## 故障处理

申请保存成功不等于邮件发出。前端只确认申请登记，后台独立显示通知状态。未配置通知时每60秒重查；发送失败最多尝试3次，后台可手动重试。服务重启后会继续检查数据库中的待发申请。成功状态表示SMTP服务器接受邮件，不代表用户已经阅读或收件箱已确认收到。

同一提交编号重试只创建一条申请，成功的通知不重复发送。发送过程中服务崩溃可能导致投递状态未知，后台需先核对邮箱再手动重试。邮件仅包含问题描述和申请编号，完整会话仅在有管理员权限的后台查看。

## 验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_handoff.py -q
Set-Location front
npm test
npm run build
```

离线测试隔离邮件配置，不会向真实邮箱发信。真实验收提交一条明确标注的测试申请，核对数据库、后台通知状态及实际邮箱，处理后标为已处理。
