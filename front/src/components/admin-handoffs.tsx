import { Alert, App, Button, Card, Empty, Flex, Form, Input, Select, Spin, Tag, Typography } from "antd";
import { ClipboardList, RefreshCw, Save } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchHandoff, fetchHandoffs, retryHandoffNotification, updateHandoff } from "../api/handoff";
import { getErrorMessage } from "../api/client";
import type { HandoffDetail, HandoffStatus, HandoffTicket, NotificationStatus } from "../types/handoff";

const STATUS_LABELS: Record<HandoffStatus, string> = { pending: "待处理", processing: "处理中", resolved: "已处理" };
const NOTIFICATION_LABELS: Record<NotificationStatus, string> = {
  pending: "通知待发送", sending: "通知发送中", sent: "通知已发送", failed: "通知发送失败", unconfigured: "通知未配置",
};
type HandlingForm = { status: HandoffStatus; handling_note: string };
interface AdminHandoffsProps { tenantId: string; active: boolean; onBusyChange?: (busy: boolean) => void }

function StatusTag({ status }: { status: HandoffStatus }) {
  return <Tag color={status === "resolved" ? "success" : status === "processing" ? "processing" : "warning"}>{STATUS_LABELS[status]}</Tag>;
}
function NotificationTag({ status }: { status: NotificationStatus }) {
  return <Tag color={status === "sent" ? "success" : ["failed", "unconfigured"].includes(status) ? "error" : "default"}>{NOTIFICATION_LABELS[status]}</Tag>;
}

export function AdminHandoffs({ tenantId, active, onBusyChange }: AdminHandoffsProps) {
  const { message } = App.useApp();
  const [form] = Form.useForm<HandlingForm>();
  const [list, setList] = useState<{ tenantId: string; filter: HandoffStatus | "all"; tickets: HandoffTicket[] }>({ tenantId, filter: "pending", tickets: [] });
  const [hasMore, setHasMore] = useState(false);
  const [selected, setSelected] = useState<{ tenantId: string; ticketId: string } | null>(null);
  const [detail, setDetail] = useState<HandoffDetail | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [filter, setFilter] = useState<HandoffStatus | "all">("pending");
  const listVersion = useRef(0);
  const detailVersion = useRef(0);
  const currentScope = useRef(tenantId);
  const inFlight = useRef(false);
  const currentFilter = useRef(filter);
  const nextOffset = useRef(0);
  currentScope.current = tenantId;
  currentFilter.current = filter;
  const busy = saving || retrying;

  useEffect(() => { onBusyChange?.(busy); }, [busy, onBusyChange]);

  const loadList = useCallback(async (append = false) => {
    const version = ++listVersion.current;
    const offset = append ? nextOffset.current : 0;
    const isCurrent = () => version === listVersion.current && currentScope.current === tenantId && currentFilter.current === filter;
    setLoadingList(true);
    setListError(null);
    try {
      const tickets = await fetchHandoffs(tenantId, filter === "all" ? undefined : filter, offset, 50);
      if (!isCurrent()) return;
      nextOffset.current = offset + tickets.length;
      setHasMore(tickets.length === 50);
      setList((previous) => {
        const previousTickets = append && previous.tenantId === tenantId && previous.filter === filter ? previous.tickets : [];
        const merged = new Map([...previousTickets, ...tickets].map((ticket) => [ticket.ticket_id, ticket]));
        return { tenantId, filter, tickets: [...merged.values()] };
      });
    } catch (error) {
      if (isCurrent()) setListError(getErrorMessage(error, "加载人工待办失败"));
    } finally {
      if (isCurrent()) setLoadingList(false);
    }
  }, [tenantId, filter]);

  useEffect(() => {
    if (!active) return;
    nextOffset.current = 0;
    setHasMore(false);
    setList({ tenantId, filter, tickets: [] });
    setSelected(null);
    setDetail(null);
    setDetailError(null);
    void loadList();
    return () => { ++listVersion.current; ++detailVersion.current; };
  }, [active, tenantId, loadList]);

  useEffect(() => {
    if (!selected || selected.tenantId !== tenantId) return;
    const version = ++detailVersion.current;
    setDetail(null);
    setDetailError(null);
    setLoadingDetail(true);
    void fetchHandoff(selected.ticketId, tenantId).then((result) => {
      if (version !== detailVersion.current || currentScope.current !== tenantId) return;
      setDetail(result);
      form.setFieldsValue({ status: result.status, handling_note: result.handling_note || "" });
    }).catch((error) => {
      if (version === detailVersion.current && currentScope.current === tenantId) setDetailError(getErrorMessage(error, "加载事项详情失败"));
    }).finally(() => {
      if (version === detailVersion.current && currentScope.current === tenantId) setLoadingDetail(false);
    });
    return () => { ++detailVersion.current; };
  }, [selected, tenantId, form]);

  useEffect(() => {
    if (!active || !detail || !["pending", "sending"].includes(detail.notification_status)) return;
    const ticketId = detail.ticket_id;
    let cancelled = false;
    let checking = false;
    const checkNotification = async () => {
      if (checking) return;
      checking = true;
      try {
        const result = await fetchHandoff(ticketId, tenantId);
        if (cancelled || currentScope.current !== tenantId) return;
        setDetail((current) => current?.ticket_id === ticketId ? {
          ...current, notification_status: result.notification_status, notification_error: result.notification_error,
        } : current);
        setList((current) => current.tenantId !== tenantId ? current : {
          ...current, tickets: current.tickets.map((ticket) => ticket.ticket_id === ticketId ? {
            ...ticket, notification_status: result.notification_status, notification_error: result.notification_error,
          } : ticket),
        });
      } catch { /* The refresh and detail reload actions remain available. */ }
      finally { checking = false; }
    };
    void checkNotification();
    const timer = window.setInterval(() => { void checkNotification(); }, 3000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active, tenantId, detail?.ticket_id, detail?.notification_status]);

  const applyDetail = (result: HandoffDetail, resetForm = true) => {
    if (currentScope.current !== tenantId) return;
    setDetail(result);
    if (resetForm) form.setFieldsValue({ status: result.status, handling_note: result.handling_note || "" });
    setList((value) => value.tenantId !== tenantId ? value : {
      tenantId, filter: value.filter, tickets: value.tickets.map((ticket) => ticket.ticket_id === result.ticket_id ? result : ticket),
    });
  };
  const saveHandling = async ({ status, handling_note }: HandlingForm) => {
    if (!detail || inFlight.current) return;
    inFlight.current = true;
    setSaving(true);
    setDetailError(null);
    try {
      applyDetail(await updateHandoff(detail.ticket_id, tenantId, status, handling_note || ""));
      message.success("处理结果已保存");
      void loadList();
    } catch (error) { setDetailError(getErrorMessage(error, "保存处理结果失败")); }
    finally { inFlight.current = false; setSaving(false); }
  };
  const retryNotification = async () => {
    if (!detail || inFlight.current) return;
    inFlight.current = true;
    setRetrying(true);
    setDetailError(null);
    try {
      const result = await retryHandoffNotification(detail.ticket_id, tenantId);
      applyDetail({ ...detail, notification_status: result.notification_status, notification_error: null }, false);
      if (result.notification_status === "sent") message.success("邮件通知已发送");
      else if (["failed", "unconfigured"].includes(result.notification_status)) message.warning("申请仍已保存，通知暂未发送");
      else message.success("已提交通知重试");
    } catch (error) { setDetailError(getErrorMessage(error, "重试邮件通知失败")); }
    finally { inFlight.current = false; setRetrying(false); }
  };
  const tickets = list.tenantId === tenantId && list.filter === filter ? list.tickets : [];
  const filteredTickets = tickets.filter((ticket) => filter === "all" || ticket.status === filter);
  const showDetail = selected?.tenantId === tenantId;

  return <Flex vertical gap={16}>
    <Flex justify="space-between" align="center" gap={12} wrap>
      <Flex align="center" gap={10}>
        <Typography.Text strong>人工待办</Typography.Text>
        <Select aria-label="筛选处理状态" value={filter} onChange={setFilter} disabled={busy} style={{ width: 120 }} options={[
          { value: "pending", label: "待处理" }, { value: "processing", label: "处理中" },
          { value: "resolved", label: "已处理" }, { value: "all", label: "全部" },
        ]} />
      </Flex>
      <Button icon={<RefreshCw size={15} />} loading={loadingList} disabled={busy} onClick={() => { void loadList(); }}>刷新待办</Button>
    </Flex>
    {listError && <Alert showIcon type="error" title={listError} />}
    <div className="handoff-admin-grid">
      <Spin spinning={loadingList}>
        <Flex vertical gap={12}>
          {!filteredTickets.length ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={listError ? "暂时无法加载事项，请刷新重试" : "暂无该状态的事项"} />
            : filteredTickets.map((ticket) => <Card key={ticket.ticket_id} size="small">
              <Flex vertical gap={10}>
                <Typography.Paragraph className="handoff-issue" ellipsis={{ rows: 3 }}>{ticket.issue}</Typography.Paragraph>
                <Flex wrap gap={6}><StatusTag status={ticket.status} /><NotificationTag status={ticket.notification_status} /></Flex>
                <Typography.Text type="secondary" className="handoff-meta">{new Date(ticket.created_at).toLocaleString("zh-CN")}</Typography.Text>
                <Button icon={<ClipboardList size={15} />} disabled={busy}
                  aria-label={`查看事项 ${ticket.ticket_id}`}
                  onClick={() => setSelected({ tenantId, ticketId: ticket.ticket_id })}>查看并处理</Button>
              </Flex>
            </Card>)}
          {hasMore && <Button loading={loadingList} disabled={busy} onClick={() => { void loadList(true); }}>加载更多事项</Button>}
        </Flex>
      </Spin>
      {showDetail && <Spin spinning={loadingDetail}>
        <Card size="small" title="事项详情">
          {detailError && <Alert showIcon type="error" title={detailError} className="admin-alert" />}
          {!detail && !loadingDetail && <Button onClick={() => setSelected({ ...selected! })}>重新加载详情</Button>}
          {detail && <>
            <Typography.Text type="secondary">申请编号：{detail.ticket_id}</Typography.Text>
            <Typography.Paragraph className="handoff-issue">{detail.issue}</Typography.Paragraph>
            <Flex wrap gap={6} className="file-tags"><StatusTag status={detail.status} /><NotificationTag status={detail.notification_status} /></Flex>
            {detail.notification_error && <Alert showIcon type="warning" title={detail.notification_error} className="admin-alert" />}
            {["failed", "unconfigured"].includes(detail.notification_status) &&
              <Button loading={retrying} disabled={saving} onClick={retryNotification} className="admin-alert">重试邮件通知</Button>}
            <Form form={form} layout="vertical" onFinish={saveHandling} disabled={busy} requiredMark={false}>
              <Form.Item name="status" label="处理状态" rules={[{ required: true }]}>
                <Select options={Object.entries(STATUS_LABELS).map(([value, label]) => ({ value, label }))} />
              </Form.Item>
              <Form.Item name="handling_note" label="处理备注" rules={[{ max: 4000, message: "最多填写 4000 个字符" }]}>
                <Input.TextArea rows={3} maxLength={4000} placeholder="记录核查情况或处理结果" />
              </Form.Item>
              <Button type="primary" htmlType="submit" loading={saving} disabled={retrying} icon={<Save size={15} />}>保存处理结果</Button>
            </Form>
            <Typography.Title level={5}>相关会话</Typography.Title>
            {!detail.messages.length
              ? <Typography.Text type="secondary">暂无已保存的会话记录</Typography.Text>
              : <Flex vertical gap={12}>{detail.messages.map((item, index) => <div key={index}>
                <Typography.Text strong>{item.role === "user" ? "用户" : item.role === "assistant" ? "AI 客服" : "系统"}</Typography.Text>
                <Typography.Paragraph className="handoff-context">{item.content}</Typography.Paragraph>
              </div>)}</Flex>}
          </>}
        </Card>
      </Spin>}
    </div>
  </Flex>;
}
