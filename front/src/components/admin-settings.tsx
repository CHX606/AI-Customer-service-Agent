import { Button, Flex, Input, Modal, Tabs, Typography } from "antd";
import { Building2, ClipboardList, Database, RefreshCw, Save } from "lucide-react";
import { useAdminSettings } from "../hooks/use-admin-settings";
import { AdminHandoffs } from "./admin-handoffs";
import { AdminProfile } from "./admin-profile";
import { AdminKnowledge } from "./admin-knowledge";

interface AdminSettingsProps { isOpen: boolean; onClose: () => void; onProfileUpdated?: () => void }
function ProfileFooter({ profile }: { profile: ReturnType<typeof useAdminSettings>["profile"] }) {
  return <Flex justify="space-between" align="center" gap={12}>
    <Button onClick={profile.loadProfileData} icon={<RefreshCw size={15} />} disabled={profile.isLoadingProfile || profile.isSavingProfile}>重新加载</Button>
    <Button type="primary" htmlType="submit" form="company-profile-form" loading={profile.isSavingProfile}
      disabled={profile.isLoadingProfile} icon={<Save size={16} />}>保存配置</Button>
  </Flex>;
}
export function AdminSettings({ isOpen, onClose, onProfileUpdated }: AdminSettingsProps) {
  const state = useAdminSettings(isOpen, onProfileUpdated);
  return <Modal open={isOpen} onCancel={onClose} centered width={920} className="admin-modal"
    title={<Flex align="center" gap={10}><Building2 size={22} />企业客服管理</Flex>}
    styles={{ body: { display: "flex", flexDirection: "column", maxHeight: "min(68dvh, 720px)", overflow: "hidden" } }}
    footer={state.activeTab === "profile" ? <ProfileFooter profile={state.profile} /> : null}>
    <Typography.Paragraph type="secondary" className="admin-description">维护企业资料与知识库，查看并处理用户提交的事项。</Typography.Paragraph>
    <Flex align="center" gap={12} className="admin-toolbar">
      <Typography.Text><label htmlFor="tenant-id">企业标识</label></Typography.Text>
      <Input id="tenant-id" value={state.tenantId} onChange={event => state.setTenantId(event.target.value)} placeholder="default"
        disabled={state.tenantDisabled} className="tenant-input" />
    </Flex>
    <Tabs activeKey={state.activeTab} onChange={state.setActiveTab} className="admin-tabs" animated={{ inkBar: true, tabPane: true }} items={[
      { key: "profile", label: "企业资料", icon: <Building2 size={16} />, children: <AdminProfile profile={state.profile} profileForm={state.profileForm} /> },
      { key: "knowledge", label: "知识库文档", icon: <Database size={16} />, children: <AdminKnowledge knowledge={state.knowledge} /> },
      { key: "handoffs", label: "人工待办", icon: <ClipboardList size={16} />, children: <AdminHandoffs tenantId={state.tenantId}
        active={isOpen && state.activeTab === "handoffs"} onBusyChange={state.setHandoffBusy} /> },
    ]} />
  </Modal>;
}
