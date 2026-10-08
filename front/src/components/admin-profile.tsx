import { Alert, Form, Input, Spin, Typography, type FormInstance } from "antd";
import type { ProfileFormState } from "../hooks/use-admin-profile";
import type { useAdminProfile } from "../hooks/use-admin-profile";

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

interface AdminProfileProps { profile: ReturnType<typeof useAdminProfile>; profileForm: FormInstance<ProfileFormState> }
export function AdminProfile({ profile, profileForm }: AdminProfileProps) {
  return <Spin spinning={profile.isLoadingProfile}>
    <Form form={profileForm} id="company-profile-form" layout="vertical" onFinish={profile.handleSaveProfile}
      initialValues={profile.form} disabled={profile.isLoadingProfile || profile.isSavingProfile} requiredMark="optional"
      scrollToFirstError={{ block: "center", behavior: "smooth" }} className="admin-form">
      {profile.profileMsg && <Alert showIcon type={profile.profileMsg.type} title={profile.profileMsg.text} className="admin-alert" />}
      {FIELD_GROUPS.map(group => <section key={group.title} className="admin-field-section">
        <Typography.Title level={5}>{group.title}</Typography.Title>
        <div className="admin-form-grid">{group.fields.map(field => <Form.Item key={field.name} name={field.name} label={field.label}
          className={field.wide ? "admin-field-wide" : undefined}
          rules={field.required ? [{ required: true, whitespace: true, message: `请填写${field.label}` }] : undefined}>
          {field.rows ? <Input.TextArea rows={field.rows} placeholder={field.placeholder} /> : <Input placeholder={field.placeholder} />}
        </Form.Item>)}</div>
      </section>)}
    </Form>
  </Spin>;
}
