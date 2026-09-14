import { useState } from "react";
import { fetchAdminProfile, updateAdminProfile } from "../api/profile";
import { getErrorMessage } from "../api/client";
import type { TenantProfileUpdate } from "../types/profile";

export interface ProfileFormState {
  companyName: string;
  brandNameEn: string;
  assistantName: string;
  shortDescription: string;
  businessScopeText: string;
  businessHours: string;
  publicContact: string;
  welcomeTitle: string;
  welcomeDescription: string;
  tone: string;
  handoffMessage: string;
  suggestedQuestionsText: string;
}

const EMPTY_PROFILE_FORM: ProfileFormState = {
  companyName: "",
  brandNameEn: "",
  assistantName: "",
  shortDescription: "",
  businessScopeText: "",
  businessHours: "",
  publicContact: "",
  welcomeTitle: "",
  welcomeDescription: "",
  tone: "",
  handoffMessage: "",
  suggestedQuestionsText: "",
};


export function useAdminProfile(tenantId: string, onProfileUpdated?: () => void) {
  const [isLoadingProfile, setIsLoadingProfile] = useState(false);

  const [isSavingProfile, setIsSavingProfile] = useState(false);

  const [profileMsg, setProfileMsg] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  const [form, setForm] = useState<ProfileFormState>(EMPTY_PROFILE_FORM);

  const loadProfileData = async () => {
    setIsLoadingProfile(true);
    setProfileMsg(null);
    try {
      const data = await fetchAdminProfile(tenantId);
      setForm({
        companyName: data.company_name,
        brandNameEn: data.brand_name_en || "",
        assistantName: data.assistant_name,
        shortDescription: data.short_description,
        businessScopeText: data.business_scope.join("\n"),
        businessHours: data.business_hours || "",
        publicContact: data.public_contact || "",
        welcomeTitle: data.welcome_title,
        welcomeDescription: data.welcome_description,
        tone: data.tone,
        handoffMessage: data.handoff_message,
        suggestedQuestionsText: data.suggested_questions.join("\n"),
      });
    } catch (error: unknown) {
      setProfileMsg({
        type: "error",
        text: getErrorMessage(error, "加载配置失败"),
      });
    } finally {
      setIsLoadingProfile(false);
    }
  };

  const handleSaveProfile = async (values: ProfileFormState) => {
    setIsSavingProfile(true);
    setProfileMsg(null);

    const updatePayload: TenantProfileUpdate = {
      tenant_id: tenantId,
      company_name: values.companyName,
      brand_name_en: values.brandNameEn || undefined,
      assistant_name: values.assistantName,
      short_description: values.shortDescription,
      business_scope: values.businessScopeText
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      business_hours: values.businessHours || undefined,
      public_contact: values.publicContact || undefined,
      welcome_title: values.welcomeTitle,
      welcome_description: values.welcomeDescription,
      tone: values.tone,
      handoff_message: values.handoffMessage,
      suggested_questions: values.suggestedQuestionsText
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
    };

    try {
      await updateAdminProfile(updatePayload);
      setProfileMsg({
        type: "success",
        text: "企业资料已成功保存！前端已同步更新。",
      });
      onProfileUpdated?.();
    } catch (error: unknown) {
      setProfileMsg({
        type: "error",
        text: getErrorMessage(error, "保存企业资料失败"),
      });
    } finally {
      setIsSavingProfile(false);
    }
  };
  return { isLoadingProfile, isSavingProfile, profileMsg, form, loadProfileData, handleSaveProfile };
}
