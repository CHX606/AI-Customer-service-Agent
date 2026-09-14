"""默认租户（可乐云）的预设配置数据。

未经过正式确认的企业客服时间与公开联系方式保持为空（None），
由管理员在后台根据企业真实情况填写。
"""

from datetime import datetime
from back.domain.tenant import TenantProfile


DEFAULT_KELECLOUD_PROFILE = TenantProfile(
    tenant_id="default",
    company_name="可乐云",
    brand_name_en="KeleCloud",
    assistant_name="可乐云 AI 智能客服",
    short_description=(
        "可乐云是一家网络连接与加速服务提供商，"
        "致力于为用户提供网络节点服务与客户端使用支持。"
    ),
    business_scope=[
        "账号注册与登录管理",
        "套餐购买与订单支付",
        "套餐续费与流量重置",
        "订阅链接获取与更新",
        "节点连接与网络异常排查",
        "客户端软件下载、安装与配置",
        "X/Twitter、Google、YouTube、Telegram等外网应用的连接、登录与基础排障",
        "退款政策与售后服务咨询",
    ],
    business_hours=None,  # 未确认字段保持为空，由后台配置
    public_contact=None,  # 未确认字段保持为空，由后台配置
    welcome_title="你好，我是可乐云智能客服",
    welcome_description="随时向我咨询业务规则、账号异常排查，支持直接粘贴或上传故障截图进行智能分析。",
    tone="简洁、友好、专业",
    handoff_message="抱歉，目前知识库中暂未查到能够明确回答该问题的资料，建议联系人工客服进一步处理。",
    suggested_questions=[
        "续费之后为什么流量没有重置？",
        "忘记账号或密码应该怎么办？",
        "软件突然不能使用了，应该如何排查？",
        "如何更新订阅或重新导入节点？",
        "怎么登录X（Twitter）？",
    ],
    created_at=datetime.now(),
    updated_at=datetime.now(),
)
