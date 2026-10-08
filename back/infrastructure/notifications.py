"""应用内 SMTP 通知适配器；配置和邮件内容不进入公开接口或日志。"""
import json
import os
import re
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate
from urllib.parse import urlparse

from back.domain.handoff import HandoffTicket


def _local_time(value: str) -> str:
    """邮件里按服务器系统时区显示提交时间；解析不了时原样显示。"""
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return value
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone().strftime("%Y-%m-%d %H:%M:%S")


class NotificationConfigurationError(ValueError):
    """消息只能包含可安全展示的配置原因。"""


@dataclass(frozen=True)
class EmailSettings:
    host: str
    port: int
    username: str
    password: str = field(repr=False)
    sender: str
    recipient: str
    security: str
    admin_url: str
    timeout: float = 15


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    if "\r" in value or "\n" in value:
        raise NotificationConfigurationError("邮件配置包含非法换行。")
    return value.strip()


def _address(value: str) -> str:
    if not re.fullmatch(r"[^@\s<>;,]+@[^@\s<>;,]+\.[^@\s<>;,]+", value):
        raise NotificationConfigurationError("邮件地址配置无效。")
    return value


def _settings(tenant_id: str) -> EmailSettings:
    host = _setting("SMTP_HOST")
    username = _setting("SMTP_USERNAME")
    # Preserve non-Gmail passwords exactly, including intentional internal/outer spaces.
    password = os.getenv("SMTP_PASSWORD", "")
    if "\r" in password or "\n" in password:
        raise NotificationConfigurationError("邮件配置包含非法换行。")
    if host.lower() == "smtp.gmail.com":
        password = "".join(password.split())
    sender = _setting("SMTP_FROM") or username
    security = _setting("SMTP_SECURITY", "starttls").lower()
    recipients = _setting("HANDOFF_NOTIFICATION_RECIPIENTS")
    recipient = ""
    if recipients:
        try:
            mapping = json.loads(recipients)
            if not isinstance(mapping, dict):
                raise ValueError()
            recipient = mapping.get(tenant_id, "")
            if not isinstance(recipient, str):
                raise ValueError()
            recipient = recipient.strip()
            if "\r" in recipient or "\n" in recipient:
                raise ValueError()
        except (ValueError, TypeError):
            raise NotificationConfigurationError("人工通知收件人映射配置无效。") from None
    if not recipient and tenant_id == "default":
        recipient = _setting("HANDOFF_NOTIFICATION_TO")
    if not host or not sender or not recipient:
        raise NotificationConfigurationError("邮件通知尚未配置完整。")
    if bool(username) != bool(password):
        raise NotificationConfigurationError("邮件通知账号或密码尚未配置完整。")
    if security not in {"ssl", "starttls"}:
        raise NotificationConfigurationError("SMTP_SECURITY 必须为 ssl 或 starttls。")
    try:
        port = int(_setting("SMTP_PORT", "465" if security == "ssl" else "587"))
        timeout = float(_setting("SMTP_TIMEOUT_SECONDS", "15"))
        if not 1 <= port <= 65535 or not 1 <= timeout <= 30:
            raise ValueError()
    except ValueError:
        raise NotificationConfigurationError("SMTP端口或超时配置无效。") from None
    admin_url = _setting("HANDOFF_ADMIN_URL")
    if admin_url:
        try:
            parsed = urlparse(admin_url)
            valid = parsed.scheme in {"https", "http"} and parsed.hostname and not parsed.username and not parsed.password
        except ValueError:
            valid = False
        if not valid:
            raise NotificationConfigurationError("后台地址必须是有效的HTTP或HTTPS地址。")
    return EmailSettings(host, port, username, password, _address(sender), _address(recipient),
                         security, admin_url, timeout)


class EmailNotifier:
    def availability(self, tenant_id: str) -> tuple[bool, str | None]:
        try:
            _settings(tenant_id)
        except NotificationConfigurationError as exc:
            return False, str(exc)
        return True, None

    def send(self, ticket: HandoffTicket) -> None:
        settings = _settings(ticket.tenant_id)
        message = EmailMessage()
        message["Subject"] = f"人工处理申请 {ticket.ticket_id}"
        message["From"] = settings.sender
        message["To"] = settings.recipient
        message["Date"] = formatdate(localtime=False)
        message["Message-ID"] = f"<handoff.{ticket.ticket_id}@{settings.sender.rsplit('@', 1)[1]}>"
        body = (f"收到新的人工处理申请。\n\n申请编号：{ticket.ticket_id}\n租户：{ticket.tenant_id}\n"
                f"提交时间：{_local_time(ticket.created_at)}\n\n用户描述：\n{ticket.issue}\n\n"
                "请登录后台查看会话记录并处理。\n")
        if settings.admin_url:
            body += f"后台地址：{settings.admin_url}\n"
        message.set_content(body)
        context = ssl.create_default_context()
        if settings.security == "ssl":
            transport = smtplib.SMTP_SSL(settings.host, settings.port, timeout=settings.timeout, context=context)
        else:
            transport = smtplib.SMTP(settings.host, settings.port, timeout=settings.timeout)
        with transport as connection:
            if settings.security == "starttls":
                connection.ehlo()
                connection.starttls(context=context)
                connection.ehlo()
            if settings.username:
                connection.login(settings.username, settings.password)
            refused = connection.send_message(message, from_addr=settings.sender, to_addrs=[settings.recipient])
            if refused:
                raise smtplib.SMTPRecipientsRefused(refused)
