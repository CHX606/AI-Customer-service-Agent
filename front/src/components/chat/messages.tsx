import { ActionBarPrimitive, MessagePartPrimitive, MessagePrimitive, useAuiState, type Attachment } from "@assistant-ui/react";
import { Bubble } from "@ant-design/x";
import { Avatar, Button, Flex, Image, Tag, Tooltip, Typography } from "antd";
import { Bot, Check, Copy, ImagePlus, RotateCcw, UserRound } from "lucide-react";
import { useContext, useEffect, useState } from "react";
import { ProfileContext } from "./profile-context";

function UserMessageAttachment({ attachment }: { attachment: Attachment }) {
  const [url, setUrl] = useState<string>();
  useEffect(() => {
    if (!attachment.file) { setUrl(undefined); return; }
    const objectUrl = URL.createObjectURL(attachment.file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [attachment.file]);
  if (attachment.type !== "image") return null;
  return <div className="message-attachment">{url
    ? <Image src={url} alt={attachment.name || "用户上传截图"} width={180} styles={{ image: { maxHeight: 150, objectFit: "cover", borderRadius: 8 } }} />
    : <Tag icon={<ImagePlus size={15} />}>{attachment.name}</Tag>}
  </div>;
}

function MessageText() {
  return <MessagePrimitive.Parts>{({ part }) => part.type === "text"
    ? <MessagePartPrimitive.Text className="message-text" smooth={false} />
    : null}</MessagePrimitive.Parts>;
}

export function UserMessage() {
  return <MessagePrimitive.Root className="message-root">
    <Bubble placement="end" variant="filled" shape="corner"
      avatar={<Avatar icon={<UserRound size={19} />} />}
      header={<Typography.Text type="secondary">你</Typography.Text>}
      content={<>
        <MessagePrimitive.Attachments>{({ attachment }) => <UserMessageAttachment attachment={attachment} />}</MessagePrimitive.Attachments>
        <MessageText />
      </>}
    />
  </MessagePrimitive.Root>;
}

export function AssistantMessage() {
  const profile = useContext(ProfileContext);
  const copied = useAuiState((state) => state.message.isCopied);
  const isLoading = useAuiState((state) => state.message.status?.type === "running" && state.message.parts.every((part) => part.type === "text" && !part.text?.trim()));
  return <MessagePrimitive.Root className="message-root">
    <Bubble placement="start" variant="outlined" shape="corner" loading={isLoading}
      avatar={<Avatar className="brand-avatar" icon={<Bot size={20} />} />}
      header={<Typography.Text type="secondary">{profile.assistant_name}</Typography.Text>}
      content={<MessageText />}
      footer={<ActionBarPrimitive.Root hideWhenRunning autohide="not-last">
        <Flex gap={4}>
          <Tooltip title="复制回复"><ActionBarPrimitive.Copy asChild>
            <Button type="text" size="small" icon={copied ? <Check size={15} /> : <Copy size={15} />} aria-label="复制回复">{copied ? "已复制" : "复制"}</Button>
          </ActionBarPrimitive.Copy></Tooltip>
          <Tooltip title="重新生成"><ActionBarPrimitive.Reload asChild>
            <Button type="text" size="small" icon={<RotateCcw size={15} />} aria-label="重新生成">重试</Button>
          </ActionBarPrimitive.Reload></Tooltip>
        </Flex>
      </ActionBarPrimitive.Root>}
    />
  </MessagePrimitive.Root>;
}
