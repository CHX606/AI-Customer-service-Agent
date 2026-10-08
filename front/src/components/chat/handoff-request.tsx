import { Alert, Button, Form, Input, Modal, Typography } from "antd";
import { useEffect, useRef, useState } from "react";
import { submitHandoff } from "../../api/handoff";
import type { HandoffReceipt } from "../../types/handoff";

interface HandoffRequestProps {
  open: boolean;
  sessionId: string;
  tenantId: string;
  initialIssue: string;
  onClose: () => void;
}

export function HandoffRequest({ open, sessionId, tenantId, initialIssue, onClose }: HandoffRequestProps) {
  const [form] = Form.useForm<{ issue: string }>();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [receipt, setReceipt] = useState<HandoffReceipt | null>(null);
  const draft = useRef<{ scope: string; issue: string; submissionId: string } | null>(null);
  const initializedScope = useRef<string | null>(null);
  const inFlight = useRef(false);
  const scope = `${tenantId}/${sessionId}`;

  useEffect(() => {
    if (!open || (initializedScope.current === scope && draft.current)) return;
    initializedScope.current = scope;
    draft.current = null;
    setReceipt(null);
    setError(null);
    form.setFieldsValue({ issue: initialIssue.slice(0, 2000) });
  }, [open, scope, initialIssue, form]);

  const submit = async ({ issue }: { issue: string }) => {
    if (inFlight.current || receipt) return;
    const cleanIssue = issue.trim();
    if (!cleanIssue) return;
    if (!draft.current || draft.current.scope !== scope || draft.current.issue !== cleanIssue) {
      draft.current = { scope, issue: cleanIssue, submissionId: crypto.randomUUID() };
    }
    inFlight.current = true;
    setSubmitting(true);
    setError(null);
    try {
      const result = await submitHandoff({
        tenant_id: tenantId,
        session_id: sessionId,
        submission_id: draft.current.submissionId,
        issue: cleanIssue,
      });
      setReceipt(result);
    } catch {
      setError("提交暂未确认，请重试。已填写的内容会保留，重复重试不会重复创建申请。");
    } finally {
      inFlight.current = false;
      setSubmitting(false);
    }
  };
  const close = () => { if (inFlight.current) return; if (receipt) initializedScope.current = null; onClose(); };
  const notificationUnavailable = receipt && ["failed", "unconfigured"].includes(receipt.notification_status);
  return <Modal open={open} title="提交人工处理" centered width={560}
    onCancel={close}
    keyboard={!submitting} mask={{ closable: !submitting }}
    closable={!submitting}
    footer={receipt
      ? <Button type="primary" onClick={close}>完成</Button>
      : <>
        <Button disabled={submitting} onClick={onClose}>取消</Button>
        <Button type="primary" loading={submitting} onClick={() => form.submit()}>提交申请</Button>
      </>}>
    {receipt ? <Alert showIcon type={notificationUnavailable ? "warning" : "success"}
      title={notificationUnavailable ? "申请已保存，通知暂未发送" : "申请已记录"}
      description={<>
        <Typography.Paragraph>申请编号：{receipt.ticket_id}</Typography.Paragraph>
        <Typography.Text>站长可在后台查看并处理这件事。</Typography.Text>
      </>} />
      : <>
        <Typography.Paragraph type="secondary">说明需要处理的事情，站长会在后台查看。请勿填写密码或验证码。</Typography.Paragraph>
        {error && <Alert showIcon type="error" title={error} className="admin-alert" />}
        <Form form={form} layout="vertical" requiredMark={false} onFinish={submit} disabled={submitting}>
          <Form.Item name="issue" label="需要处理的事情" rules={[
            { required: true, whitespace: true, message: "请填写需要处理的事情" },
            { max: 2000, message: "最多填写 2000 个字符" },
          ]}>
            <Input.TextArea autoFocus rows={5} maxLength={2000} showCount
              placeholder="例如：订单付款后没有到账，需要帮忙核查" />
          </Form.Item>
        </Form>
      </>}
  </Modal>;
}
