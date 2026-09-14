import { useEffect, useState } from "react";
import { Alert, Button, Card, Empty, Flex, Form, Input, Modal, Space, Spin, Tabs, Tag, Tooltip, Typography, Upload } from "antd";
import { Building2, Database, FileText, RefreshCw, RotateCcw, Save, Trash2, Upload as UploadIcon } from "lucide-react";
import { useAdminProfile, type ProfileFormState } from "../hooks/use-admin-profile";
import { useKnowledgeSources } from "../hooks/use-knowledge-sources";
import { DEFAULT_TENANT_ID } from "../api/profile";

interface AdminSettingsProps { isOpen: boolean; onClose: () => void; onProfileUpdated?: () => void }
type Field = { name: keyof ProfileFormState; label: string; required?: boolean; placeholder?: string; rows?: number; wide?: boolean };
const FIELD_GROUPS: { title: string; fields: Field[] }[] = [
  { title: "基础资料", fields: [
    { name: "companyName", label: "企业全称", required: true },
    { name: "brandNameEn", label: "英文品牌名", placeholder: "例如 KeleCloud" },
    { name: "assistantName", label: "客服助理名称", required: true },
    { name: "tone", label: "客服语气", placeholder: "简洁、友好、专业" },
    { name: "shortDescription", label: "企业简介与产品定位", required: true, rows: 3, wide: true, placeholder: "介绍企业的核心服务与定位" },
  ] },
  { title: "接待与服务", fields: [
    { name: "businessHours", label: "人工客服时间", placeholder: "例如 每天 09:00–22:00" },
    { name: "publicContact", label: "公开联系方式", placeholder: "在线工单或客服邮箱" },
    { name: "businessScopeText", label: "业务服务范围", rows: 3, wide: true, placeholder: "每行填写一项服务" },
    { name: "handoffMessage", label: "转人工提示语", wide: true },
  ] },
  { title: "欢迎与引导", fields: [
    { name: "welcomeTitle", label: "欢迎语标题", required: true, wide: true },
    { name: "welcomeDescription", label: "欢迎语说明", required: true, rows: 2, wide: true },
    { name: "suggestedQuestionsText", label: "推荐问题", rows: 4, wide: true, placeholder: "每行填写一个问题" },
  ] },
];

export function AdminSettings({ isOpen, onClose, onProfileUpdated }: AdminSettingsProps) {
  const [activeTab, setActiveTab] = useState("profile");
  const [tenantId, setTenantId] = useState(DEFAULT_TENANT_ID);
  const [profileForm] = Form.useForm<ProfileFormState>();
  const { isLoadingProfile, isSavingProfile, profileMsg, form, loadProfileData, handleSaveProfile } = useAdminProfile(tenantId, onProfileUpdated);
  const { files, isLoadingFiles, isUploading, uploadError, operatingId, loadKnowledgeData, handleFileUpload, handleDeleteFile, handleReindexFile } = useKnowledgeSources(tenantId);

  useEffect(() => {
    if (!isOpen) return;
    if (activeTab === "profile") void loadProfileData();
    else void loadKnowledgeData();
  }, [isOpen, activeTab, tenantId]);

  useEffect(() => { profileForm.setFieldsValue(form); }, [form, profileForm]);

  const profileContent = (
    <Spin spinning={isLoadingProfile}>
      <Form form={profileForm} id="company-profile-form" layout="vertical" onFinish={handleSaveProfile}
        initialValues={form} disabled={isLoadingProfile || isSavingProfile} requiredMark="optional"
        scrollToFirstError={{ block: "center", behavior: "smooth" }} className="admin-form">
        {profileMsg && <Alert showIcon type={profileMsg.type} title={profileMsg.text} className="admin-alert" />}
        {FIELD_GROUPS.map((group) => <section key={group.title} className="admin-field-section">
          <Typography.Title level={5}>{group.title}</Typography.Title>
          <div className="admin-form-grid">
            {group.fields.map((field) => <Form.Item key={field.name} name={field.name} label={field.label}
              className={field.wide ? "admin-field-wide" : undefined}
              rules={field.required ? [{ required: true, whitespace: true, message: `请填写${field.label}` }] : undefined}>
              {field.rows ? <Input.TextArea rows={field.rows} placeholder={field.placeholder} />
                : <Input placeholder={field.placeholder} />}
            </Form.Item>)}
          </div>
        </section>)}
      </Form>
    </Spin>
  );
  const knowledgeContent = (
    <Flex vertical gap={20} className="knowledge-content">
      <Upload.Dragger name="file" accept=".docx,.pdf,.txt,.md,.markdown" multiple={false}
        showUploadList={false} disabled={isUploading} beforeUpload={(file) => { void handleFileUpload(file); return false; }}>
        <Flex vertical align="center" gap={8}>
          {isUploading ? <Spin /> : <UploadIcon size={28} />}
          <Typography.Text strong>{isUploading ? "正在上传并处理文档…" : "点击选择文档，或将文件拖到这里"}</Typography.Text>
          <Typography.Text type="secondary">DOCX、PDF、TXT、Markdown · 单文件不超过 20MB</Typography.Text>
        </Flex>
      </Upload.Dragger>
      <Typography.Text type="secondary" className="knowledge-hint">当前支持文档正文，文档中的图片暂不参与回答。</Typography.Text>
      {uploadError && <Alert type="error" showIcon title={uploadError} />}
      <Flex justify="space-between" align="center" gap={12}>
        <Typography.Text strong>知识库文档 <Typography.Text type="secondary">({files.length})</Typography.Text></Typography.Text>
        <Button icon={<RefreshCw size={15} />} loading={isLoadingFiles} onClick={loadKnowledgeData}>刷新</Button>
      </Flex>
      <Spin spinning={isLoadingFiles}>
        {files.length === 0
          ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={uploadError ? "暂时无法加载文档，请稍后重试" : "还没有文档，上传后即可用于客服回答"} />
          : <Flex vertical gap={12}>{files.map((file) => <Card key={file.source_id} size="small">
            <Flex align="start" gap={12}>
              <FileText size={22} className="file-icon" />
              <div className="file-info">
                <Typography.Text strong className="file-name">{file.original_filename}</Typography.Text>
                <Flex wrap gap={6} className="file-tags">
                  <Tag>{file.file_type.toUpperCase()}</Tag>
                  <Tag color={file.status === "ready" ? "success" : file.status === "failed" ? "error" : "processing"}>
                    {{ ready: "可用于回答", failed: "处理失败", processing: "处理中", pending: "等待处理" }[file.status]}
                  </Tag>
                </Flex>
                <Typography.Text type="secondary" className="file-meta">{file.chunk_count} 个内容片段 · 更新于 {new Date(file.updated_at).toLocaleString("zh-CN")}</Typography.Text>
                {file.error_message && <Typography.Paragraph type="danger">{file.error_message}</Typography.Paragraph>}
              </div>
              <Space size={4}>
                <Tooltip title="重新处理"><Button type="text" size="small" aria-label={`重新处理 ${file.original_filename}`}
                  icon={<RotateCcw size={16} />} disabled={Boolean(operatingId) || file.status === "processing"}
                  onClick={() => handleReindexFile(file.source_id)} /></Tooltip>
                <Tooltip title="删除文档"><Button type="text" size="small" danger aria-label={`删除 ${file.original_filename}`}
                  icon={<Trash2 size={16} />} disabled={Boolean(operatingId)} onClick={() => handleDeleteFile(file.source_id)} /></Tooltip>
              </Space>
            </Flex>
          </Card>)}</Flex>}
      </Spin>
    </Flex>
  );
  return (
    <Modal open={isOpen} onCancel={onClose} centered width={920} className="admin-modal"
      title={<Flex align="center" gap={10}><Building2 size={22} />企业客服管理</Flex>}
      styles={{ body: { display: "flex", flexDirection: "column", maxHeight: "min(68dvh, 720px)", overflow: "hidden" } }}
      footer={activeTab === "profile" ? <Flex justify="space-between" align="center" gap={12}>
        <Button onClick={loadProfileData} icon={<RefreshCw size={15} />} disabled={isLoadingProfile || isSavingProfile}>重新加载</Button>
        <Button type="primary" htmlType="submit" form="company-profile-form" loading={isSavingProfile}
          disabled={isLoadingProfile} icon={<Save size={16} />}>保存配置</Button>
      </Flex> : null}>
      <Typography.Paragraph type="secondary" className="admin-description">维护企业资料与知识库，让客服回答更准确。</Typography.Paragraph>
      <Flex align="center" gap={12} className="admin-toolbar">
        <Typography.Text><label htmlFor="tenant-id">企业标识</label></Typography.Text>
        <Input id="tenant-id" value={tenantId} onChange={(event) => setTenantId(event.target.value)}
          placeholder="default" disabled={isSavingProfile || isUploading || Boolean(operatingId)} className="tenant-input" />
      </Flex>
      <Tabs activeKey={activeTab} onChange={setActiveTab} className="admin-tabs" animated={{ inkBar: true, tabPane: true }}
        items={[
          { key: "profile", label: "企业资料", icon: <Building2 size={16} />, children: profileContent },
          { key: "knowledge", label: "知识库文档", icon: <Database size={16} />, children: knowledgeContent },
        ]} />
    </Modal>
  );
}
