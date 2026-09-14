import { AssistantRuntimeProvider, SimpleImageAttachmentAdapter, ThreadPrimitive, useAuiState, useLocalRuntime } from "@assistant-ui/react";
import { Conversations } from "@ant-design/x";
import { App, Avatar, Badge, Button, Divider, Drawer, Dropdown, Flex, Form, Grid, Input, Modal, Spin, Tooltip, Typography, type InputRef } from "antd";
import { ArrowDown, Cloud, Download, Menu, MessageCircleMore, Moon, MoreHorizontal, Plus, Settings, SquarePen, Sun, Trash2 } from "lucide-react";
import { lazy, Suspense, useRef, useState } from "react";
import { useBackendProfile } from "../hooks/use-backend-profile";
import { useConversations } from "../hooks/use-conversations";
import { chatAdapter } from "../lib/chat-adapter";
import { exportChatHistory, loadStoredChatMessages, MAX_CONVERSATION_TITLE_LENGTH, renameConversation, type Conversation } from "../lib/chat-storage";
import { FeatureBoundary } from "./feature-boundary";
import { Composer } from "./chat/composer";
import { UserMessage, AssistantMessage } from "./chat/messages";
import { ProfileContext } from "./chat/profile-context";
import { Welcome } from "./chat/welcome";

const AdminSettings = lazy(() => import("./admin-settings").then((module) => ({ default: module.AdminSettings })));
const INITIAL_MESSAGES = loadStoredChatMessages();
const IMAGE_ATTACHMENT_ADAPTER = new SimpleImageAttachmentAdapter();
type AppearanceProps = { theme: "light" | "dark"; onThemeToggle: () => void };

function ChatWorkspace({ theme, onThemeToggle }: AppearanceProps) {
  const { message } = App.useApp();
  const screens = Grid.useBreakpoint();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isAdminOpen, setIsAdminOpen] = useState(false);
  const [hasOpenedAdmin, setHasOpenedAdmin] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [renameTarget, setRenameTarget] = useState<Conversation | null>(null);
  const [renameForm] = Form.useForm<{ title: string }>();
  const renameInput = useRef<InputRef>(null);
  const isEmpty = useAuiState((state) => state.thread.isEmpty);
  const { profile, isBackendOnline, loadProfile } = useBackendProfile();
  const { activeId, conversations, deleteTarget, setDeleteTarget, handleNewConversation, handleSwitchConversation, handleConfirmDelete } = useConversations(setIsSidebarOpen);

  const openAdmin = () => {
    setIsSidebarOpen(false);
    setHasOpenedAdmin(true);
    setIsAdminOpen(true);
  };
  const handleExport = () => {
    if (exportChatHistory(profile.company_name)) message.success("对话记录已导出");
    else message.info("当前暂无可导出的聊天记录");
  };
  const sidebar = (
    <Flex vertical className="sidebar-content">
      <Flex align="center" gap={12} className="brand-lockup">
        <Avatar shape="square" size={42} icon={<Cloud size={23} />} className="brand-avatar" />
        <div className="brand-info">
          <Typography.Text strong>{profile.brand_name_en || profile.company_name}</Typography.Text>
          <Typography.Text type="secondary" className="brand-caption">{profile.company_name} · 客服工作台</Typography.Text>
        </div>
      </Flex>
      <Button type="primary" block icon={<Plus size={18} />} onClick={handleNewConversation}>新建对话</Button>
      <Divider className="sidebar-divider" />
      <Typography.Text type="secondary" className="sidebar-caption">历史会话</Typography.Text>
      <Conversations
        className="conversation-list"
        activeKey={activeId}
        onActiveChange={handleSwitchConversation}
        items={conversations.map((conversation) => ({ key: conversation.id, label: conversation.title, icon: <MessageCircleMore size={16} /> }))}
        menu={(item) => ({
          items: [
            { key: "rename", label: "重命名", icon: <SquarePen size={15} /> },
            { key: "delete", label: "删除会话", icon: <Trash2 size={15} />, danger: true },
          ],
          onClick: ({ key }) => {
            const target = conversations.find((conversation) => conversation.id === item.key);
            if (!target) return;
            if (key === "rename") {
              renameForm.resetFields();
              renameForm.setFieldsValue({ title: target.title });
              setRenameTarget(target);
              setIsSidebarOpen(false);
            } else if (key === "delete") setDeleteTarget(target);
          },
        })}
      />
      <Flex vertical gap={8} className="sidebar-tools">
        <Button icon={<Settings size={16} />} onClick={openAdmin} block>企业资料与知识库</Button>
        <Button type="text" icon={<Download size={16} />} onClick={handleExport} block>导出对话记录</Button>
      </Flex>
    </Flex>
  );
  return (
    <ProfileContext.Provider value={profile}>
      <main className="app-shell">
        {screens.lg && <aside className="product-sidebar" aria-label="会话导航">{sidebar}</aside>}
        <Drawer title="会话导航" placement="left" size={288} open={!screens.lg && isSidebarOpen} focusable={{ focusTriggerAfterClose: !renameTarget }}
          onClose={() => setIsSidebarOpen(false)} styles={{ body: { padding: 20 } }}>{sidebar}</Drawer>
        <section className="chat-panel" aria-label="智能客服对话">
          <header className="chat-header">
            {!screens.lg && <Button type="text" icon={<Menu size={20} />} aria-label="打开侧边菜单" onClick={() => setIsSidebarOpen(true)} />}
            <div className="chat-heading">
              <Typography.Title level={2}>{profile.assistant_name}</Typography.Title>
              <Badge status={isBackendOnline ? "success" : "default"} text={isBackendOnline ? "客服在线" : "服务暂未连接"} />
            </div>
            <Flex align="center" gap={8} className="header-actions">
              {screens.md && <Button icon={<Settings size={16} />} onClick={openAdmin}>管理后台</Button>}
              <Tooltip title={theme === "dark" ? "切换到浅色主题" : "切换到深色主题"}>
                <Button type="text" icon={theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
                  onClick={onThemeToggle} aria-label={theme === "dark" ? "切换到浅色主题" : "切换到深色主题"} aria-pressed={theme === "dark"} />
              </Tooltip>
              <Dropdown trigger={["click"]} menu={{ items: [
                ...(!screens.md ? [{ key: "admin", label: "企业资料与知识库", icon: <Settings size={16} />, onClick: openAdmin }] : []),
                { key: "export", label: "导出对话记录", icon: <Download size={16} />, onClick: handleExport },
              ] }}>
                <Button type="text" icon={<MoreHorizontal size={20} />} aria-label="更多操作" />
              </Dropdown>
            </Flex>
          </header>
          <ThreadPrimitive.Root className={`thread-root ${isEmpty ? "thread-empty" : ""}`}>
            <ThreadPrimitive.Viewport className="thread-viewport">
              {isEmpty && <Welcome profile={profile} />}
              {!isEmpty && <div className="message-list">
                <ThreadPrimitive.Messages>
                  {({ message: item }) => item.role === "user" ? <UserMessage /> : <AssistantMessage />}
                </ThreadPrimitive.Messages>
              </div>}
              <ThreadPrimitive.ViewportFooter className="thread-footer">
                <ThreadPrimitive.ScrollToBottom asChild>
                  <Button shape="circle" className="scroll-to-bottom" icon={<ArrowDown size={18} />} aria-label="滚动到最新消息" />
                </ThreadPrimitive.ScrollToBottom>
                <Composer />
              </ThreadPrimitive.ViewportFooter>
            </ThreadPrimitive.Viewport>
          </ThreadPrimitive.Root>
        </section>
        {hasOpenedAdmin && (
          <FeatureBoundary name="管理后台" onClose={() => { setIsAdminOpen(false); setHasOpenedAdmin(false); }}>
            <Suspense fallback={<Modal open={isAdminOpen} centered footer={null} title="管理后台" onCancel={() => setIsAdminOpen(false)}><Flex justify="center" className="loading-panel"><Spin /></Flex></Modal>}>
              <AdminSettings isOpen={isAdminOpen} onClose={() => setIsAdminOpen(false)} onProfileUpdated={loadProfile} />
            </Suspense>
          </FeatureBoundary>
        )}
        <Modal open={Boolean(renameTarget)} title="重命名会话" centered width={440} forceRender
          okText="保存" cancelText="取消" onCancel={() => setRenameTarget(null)}
          onOk={() => renameForm.submit()}
          afterOpenChange={(open) => { if (open) renameInput.current?.focus({ cursor: "all" }); }}>
          <Form form={renameForm} layout="vertical" requiredMark={false}
            onFinish={({ title }) => {
              if (!renameTarget) return;
              try {
                renameConversation(renameTarget.id, title);
                setRenameTarget(null);
                message.success("会话名称已更新");
              } catch (error) {
                message.error(error instanceof Error ? error.message : "重命名失败，请重试");
              }
            }}>
            <Form.Item name="title" label="会话名称" rules={[
              { required: true, whitespace: true, message: "请输入会话名称" },
              { max: MAX_CONVERSATION_TITLE_LENGTH, message: `最多输入 ${MAX_CONVERSATION_TITLE_LENGTH} 个字符` },
            ]}>
              <Input ref={renameInput} maxLength={MAX_CONVERSATION_TITLE_LENGTH} showCount placeholder="输入会话名称" />
            </Form.Item>
          </Form>
        </Modal>
        <Modal open={Boolean(deleteTarget)} title="删除会话" centered width={440}
          okText="确认删除" cancelText="取消" okButtonProps={{ danger: true }} confirmLoading={isDeleting}
          onCancel={() => { if (!isDeleting) setDeleteTarget(null); }}
          onOk={async () => {
            setIsDeleting(true);
            try { await handleConfirmDelete(); }
            catch { message.error("删除会话失败，请重试"); }
            finally { setIsDeleting(false); }
          }}>
          <Typography.Paragraph>确定删除「{deleteTarget?.title}」吗？删除后无法恢复。</Typography.Paragraph>
        </Modal>
      </main>
    </ProfileContext.Provider>
  );
}

export function CustomerServiceChat(props: AppearanceProps) {
  const runtime = useLocalRuntime(chatAdapter, { initialMessages: INITIAL_MESSAGES, adapters: { attachments: IMAGE_ATTACHMENT_ADAPTER } });
  return <AssistantRuntimeProvider runtime={runtime}><FeatureBoundary name="客服界面"><ChatWorkspace {...props} /></FeatureBoundary></AssistantRuntimeProvider>;
}
