import { useEffect, useState } from "react";
import { Form } from "antd";
import { DEFAULT_TENANT_ID } from "../api/profile";
import { useAdminProfile, type ProfileFormState } from "./use-admin-profile";
import { useKnowledgeSources } from "./use-knowledge-sources";

export function useAdminSettings(isOpen: boolean, onProfileUpdated?: () => void) {
  const [activeTab, setActiveTab] = useState("profile");
  const [tenantId, setTenantId] = useState(DEFAULT_TENANT_ID);
  const [handoffBusy, setHandoffBusy] = useState(false);
  const [profileForm] = Form.useForm<ProfileFormState>();
  const profile = useAdminProfile(tenantId, onProfileUpdated);
  const knowledge = useKnowledgeSources(tenantId);
  useEffect(() => {
    if (!isOpen) return;
    if (activeTab === "profile") void profile.loadProfileData();
    else if (activeTab === "knowledge") void knowledge.loadKnowledgeData();
  }, [isOpen, activeTab, tenantId]);
  useEffect(() => { profileForm.setFieldsValue(profile.form); }, [profile.form, profileForm]);
  const tenantDisabled = profile.isSavingProfile || knowledge.isUploading || Boolean(knowledge.operatingId) || handoffBusy;
  return { activeTab, setActiveTab, tenantId, setTenantId, profileForm, profile, knowledge, tenantDisabled, setHandoffBusy };
}
