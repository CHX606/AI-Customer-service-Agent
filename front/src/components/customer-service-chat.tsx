"use client";

import {
  AttachmentPrimitive,
  AssistantRuntimeProvider,
  AuiIf,
  ComposerPrimitive,
  MessagePrimitive,
  SimpleImageAttachmentAdapter,
  ThreadPrimitive,
  useLocalRuntime,
  type Attachment,
} from "@assistant-ui/react";
import { useEffect, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Bot,
  Check,
  Cloud,
  Database,
  Headphones,
  ImagePlus,
  MessageCircleMore,
  Plus,
  ShieldCheck,
  Sparkles,
  UserRound,
  X,
} from "lucide-react";
import { demoChatAdapter } from "../lib/demo-chat-adapter";
import {
  loadStoredChatMessages,
  resetStoredChatSession,
} from "../lib/chat-storage";


const INITIAL_MESSAGES = loadStoredChatMessages();
const IMAGE_ATTACHMENT_ADAPTER = (
  new SimpleImageAttachmentAdapter()
);


function startNewConversation() {
  const confirmed = window.confirm(
    "新建对话会清除当前浏览器中保存的聊天记录，是否继续？",
  );

  if (!confirmed) {
    return;
  }

  resetStoredChatSession();
  window.location.reload();
}

const suggestions = [
  {
    title: "续费与流量",
    prompt: "续费之后为什么流量没有重置？",
  },
  {
    title: "账号问题",
    prompt: "忘记账号或密码应该怎么办？",
  },
  {
    title: "软件无法使用",
    prompt: "软件突然不能使用了，应该如何排查？",
  },
];

function UserMessage() {
  return (
    <MessagePrimitive.Root className="message-row message-row-user">
      <div className="message-avatar message-avatar-user" aria-hidden="true">
        <UserRound size={17} strokeWidth={2} />
      </div>
      <div className="message-column message-column-user">
        <span className="message-author">你</span>
        <div className="message-bubble message-bubble-user">
          <MessagePrimitive.Parts />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="message-row message-row-assistant">
      <div
        className="message-avatar message-avatar-assistant"
        aria-hidden="true"
      >
        <Bot size={18} strokeWidth={2} />
      </div>
      <div className="message-column">
        <div className="message-author-line">
          <span className="message-author">可乐云智能客服</span>
          <span className="assistant-badge">AI</span>
        </div>
        <div className="message-bubble message-bubble-assistant">
          <MessagePrimitive.Parts />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
}

function ComposerImageAttachment({
  attachment,
}: {
  attachment: Attachment;
}) {
  const [previewUrl, setPreviewUrl] = useState<string>();

  useEffect(() => {
    if (!attachment.file) {
      setPreviewUrl(undefined);
      return undefined;
    }

    const objectUrl = URL.createObjectURL(
      attachment.file,
    );

    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [attachment.file]);

  return (
    <AttachmentPrimitive.Root className="composer-attachment">
      {previewUrl ? (
        <img
          className="composer-attachment-preview"
          src={previewUrl}
          alt="待上传截图预览"
        />
      ) : (
        <div className="composer-attachment-placeholder">
          <ImagePlus size={18} />
        </div>
      )}
      <span className="composer-attachment-name">
        <AttachmentPrimitive.Name />
      </span>
      <AttachmentPrimitive.Remove
        className="composer-attachment-remove"
        aria-label="移除图片"
      >
        <X size={15} />
      </AttachmentPrimitive.Remove>
    </AttachmentPrimitive.Root>
  );
}

function Welcome() {
  return (
    <section className="welcome" aria-labelledby="welcome-title">
      <div className="welcome-icon" aria-hidden="true">
        <Sparkles size={25} strokeWidth={1.8} />
      </div>
      <p className="welcome-eyebrow">知识库智能问答</p>
      <h1 id="welcome-title">你好，我是可乐云智能客服</h1>
      <p className="welcome-description">
        我会优先检索客服知识库，再根据相关资料回答你的问题。
      </p>

      <div className="suggestion-grid" aria-label="常见问题">
        {suggestions.map((suggestion) => (
          <ThreadPrimitive.Suggestion
            className="suggestion-card"
            key={suggestion.title}
            prompt={suggestion.prompt}
            send
          >
            <span>{suggestion.title}</span>
            <small>{suggestion.prompt}</small>
          </ThreadPrimitive.Suggestion>
        ))}
      </div>
    </section>
  );
}

function Composer() {
  return (
    <div className="composer-area">
      <ComposerPrimitive.Root className="composer-root">
        <div className="composer-attachments">
          <ComposerPrimitive.Attachments>
            {({ attachment }) => (
              <ComposerImageAttachment
                attachment={attachment}
              />
            )}
          </ComposerPrimitive.Attachments>
        </div>

        <div className="composer-input-row">
          <AuiIf
            condition={(state) => (
              state.composer.attachments.length === 0
            )}
          >
            <ComposerPrimitive.AddAttachment
              className="composer-add-image"
              aria-label="上传故障截图"
              title="上传故障截图"
              multiple={false}
            >
              <ImagePlus size={19} strokeWidth={2} />
            </ComposerPrimitive.AddAttachment>
          </AuiIf>
          <ComposerPrimitive.Input
            className="composer-input"
            aria-label="向智能客服提问"
            placeholder="描述问题，或上传故障截图"
            rows={1}
          />
          <ComposerPrimitive.Send
            className="composer-send"
            aria-label="发送消息"
          >
            <ArrowUp size={20} strokeWidth={2.3} />
          </ComposerPrimitive.Send>
        </div>
      </ComposerPrimitive.Root>
      <p className="composer-note">AI 回答可能存在误差，重要信息请以官方说明为准。</p>
    </div>
  );
}

function ChatWorkspace() {
  return (
    <main className="app-shell">
      <aside className="product-sidebar">
        <div>
          <div className="brand-lockup">
            <div className="brand-mark" aria-hidden="true">
              <Cloud size={22} strokeWidth={2} />
            </div>
            <span className="brand-console-title">Agent Console</span>
          </div>

          <div className="sidebar-section">
            <button
              className="new-conversation-button"
              type="button"
              onClick={startNewConversation}
            >
              <Plus size={18} strokeWidth={2.2} />
              <span>新对话</span>
            </button>

            <p className="sidebar-label">当前会话</p>
            <div className="conversation-item" aria-current="page">
              <MessageCircleMore size={18} />
              <span>智能客服咨询</span>
            </div>
          </div>

          <div className="sidebar-section sidebar-capabilities">
            <p className="sidebar-label">系统能力</p>
            <div className="capability-item">
              <Database size={17} />
              <span>知识库检索</span>
              <Check size={15} className="capability-check" />
            </div>
            <div className="capability-item">
              <ShieldCheck size={17} />
              <span>资料约束回答</span>
              <Check size={15} className="capability-check" />
            </div>
            <div className="capability-item">
              <Headphones size={17} />
              <span>多轮客服对话</span>
              <Check size={15} className="capability-check" />
            </div>
          </div>
        </div>

        <div className="sidebar-status">
          <span className="status-dot" aria-hidden="true" />
          <div>
            <strong>前端演示模式</strong>
            <span>尚未连接 FastAPI</span>
          </div>
        </div>
      </aside>

      <section className="chat-panel" aria-label="智能客服对话">
        <header className="chat-header">
          <div className="mobile-brand-mark" aria-hidden="true">
            <Cloud size={19} />
          </div>
          <div className="chat-heading">
            <h2>AI Customer Service Agent</h2>
            <p>
              <span className="header-status-dot" aria-hidden="true" />
              知识库服务已就绪
            </p>
          </div>
          <div className="demo-pill">
            <Sparkles size={14} />
            Demo
          </div>
        </header>

        <ThreadPrimitive.Root className="thread-root">
          <ThreadPrimitive.Viewport className="thread-viewport">
            <AuiIf condition={(state) => state.thread.isEmpty}>
              <Welcome />
            </AuiIf>

            <div className="message-list">
              <ThreadPrimitive.Messages>
                {({ message }) =>
                  message.role === "user" ? (
                    <UserMessage />
                  ) : (
                    <AssistantMessage />
                  )
                }
              </ThreadPrimitive.Messages>
            </div>

            <ThreadPrimitive.ViewportFooter className="thread-footer">
              <ThreadPrimitive.ScrollToBottom
                className="scroll-to-bottom"
                aria-label="滚动到最新消息"
              >
                <ArrowDown size={18} />
              </ThreadPrimitive.ScrollToBottom>
              <Composer />
            </ThreadPrimitive.ViewportFooter>
          </ThreadPrimitive.Viewport>
        </ThreadPrimitive.Root>
      </section>
    </main>
  );
}

export function CustomerServiceChat() {
  const runtime = useLocalRuntime(
    demoChatAdapter,
    {
      initialMessages: INITIAL_MESSAGES,
      adapters: {
        attachments: IMAGE_ATTACHMENT_ADAPTER,
      },
    },
  );

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatWorkspace />
    </AssistantRuntimeProvider>
  );
}
