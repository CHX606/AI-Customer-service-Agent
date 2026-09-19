import { useAui, useAuiState } from "@assistant-ui/react";
import { flushTapSync } from "@assistant-ui/tap";
import { Attachments, Sender } from "@ant-design/x";
import { App, Button, Tooltip, Typography, type UploadFile } from "antd";
import { ArrowUp, ImagePlus, Square } from "lucide-react";
import { useEffect, useRef, useState, type ComponentRef } from "react";

export function Composer({ imageEnabled = false }: { imageEnabled?: boolean }) {
  const aui = useAui();
  const { message } = App.useApp();
  const text = useAuiState((state) => state.composer.text);
  const attachments = useAuiState((state) => state.composer.attachments);
  const canSend = useAuiState((state) => state.composer.canSend);
  const isRunning = useAuiState((state) => state.thread.isRunning);
  const attachmentPicker = useRef<ComponentRef<typeof Attachments>>(null);
  const dropContainer = useRef<HTMLDivElement>(null);
  const addingAttachment = useRef(false);
  const [files, setFiles] = useState<UploadFile[]>([]);

  useEffect(() => {
    const urls: string[] = [];
    setFiles(attachments.map((attachment) => {
      const url = attachment.file ? URL.createObjectURL(attachment.file) : undefined;
      if (url) urls.push(url);
      return { uid: attachment.id, name: attachment.name, type: attachment.file?.type, status: attachment.status.type === "running" ? "uploading" : "done", url, thumbUrl: url };
    }));
    return () => urls.forEach(URL.revokeObjectURL);
  }, [attachments]);

  const addImage = async (file: File) => {
    if (!imageEnabled) { message.info("图片问答暂未开放，请直接输入文字问题。"); return; }
    if (isRunning || addingAttachment.current) return;
    if (!file.type.startsWith("image/")) { message.warning("请添加图片格式的截图"); return; }
    if (aui.composer.getState().attachments.length) { message.info("每次提问可附带一张截图，请先移除当前图片"); return; }
    addingAttachment.current = true;
    try { await aui.composer.addAttachment(file); }
    catch { message.error("添加截图失败，请重试"); }
    finally { addingAttachment.current = false; }
  };
  const send = () => {
    if (!imageEnabled && aui.composer.getState().attachments.length) {
      message.info("图片问答暂未开放，请先移除图片并输入文字问题。");
      return;
    }
    if (!isRunning && aui.composer.getState().canSend) aui.composer.send();
  };

  return (
    <div className="composer-area" ref={dropContainer}>
      <Sender
        aria-label="向智能客服提问"
        value={text}
        // Match assistant-ui's input: publish the controlled value before React
        // restores the DOM, including intermediate IME composition text.
        onChange={(value) => flushTapSync(() => aui.composer.setText(value))}
        onSubmit={send}
        onCancel={() => aui.thread.cancelRun()}
        loading={isRunning}
        autoSize={{ minRows: 1, maxRows: 6 }}
        placeholder={imageEnabled ? "描述你遇到的问题，或粘贴一张截图…" : "描述你遇到的问题…"}
        onPasteFile={(items) => { const image = Array.from(items).find((file) => file.type.startsWith("image/")); if (image) void addImage(image); }}
        header={imageEnabled || files.length > 0 ? <Sender.Header open={files.length > 0} forceRender title="已添加截图" closable={false}>
          <Attachments
            ref={attachmentPicker} accept="image/*" multiple={false} maxCount={1} items={files} disabled={isRunning}
            getDropContainer={() => dropContainer.current}
            beforeUpload={(file) => { void addImage(file); return false; }}
            onRemove={(file) => { void aui.composer.attachment({ id: file.uid }).remove(); return false; }}
          />
        </Sender.Header> : undefined}
        prefix={imageEnabled ? <Tooltip title="添加截图，支持拖拽和粘贴">
          <Button type="text" icon={<ImagePlus size={20} />} aria-label="上传故障截图"
            disabled={isRunning || attachments.length > 0} onClick={() => attachmentPicker.current?.select({ accept: "image/*", multiple: false })} />
        </Tooltip> : undefined}
        suffix={(_, { components: { SendButton, LoadingButton } }) => isRunning
          ? <Tooltip title="停止生成"><LoadingButton icon={<Square size={16} />} aria-label="停止生成" /></Tooltip>
          : <SendButton icon={<ArrowUp size={20} />} aria-label="发送消息" disabled={!canSend} />}
      />
      <div className="composer-note">
        <Typography.Text type="secondary">回答基于企业知识库，请核实重要信息</Typography.Text>
        <Typography.Text type="secondary" className="composer-shortcut">Enter 发送 · Shift + Enter 换行</Typography.Text>
      </div>
    </div>
  );
}
