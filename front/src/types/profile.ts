export interface TenantPublicProfile {
  tenant_id: string;
  company_name: string;
  brand_name_en?: string | null;
  assistant_name: string;
  short_description: string;
  business_scope: string[];
  business_hours?: string | null;
  public_contact?: string | null;
  welcome_title: string;
  welcome_description: string;
  tone: string;
  handoff_message: string;
  suggested_questions: string[];
  updated_at: string;
}

export interface TenantProfile extends TenantPublicProfile {
  created_at: string;
}

export interface TenantProfileUpdate {
  tenant_id: string;
  company_name?: string;
  brand_name_en?: string;
  assistant_name?: string;
  short_description?: string;
  business_scope?: string[];
  business_hours?: string;
  public_contact?: string;
  welcome_title?: string;
  welcome_description?: string;
  tone?: string;
  handoff_message?: string;
  suggested_questions?: string[];
}

export type KnowledgeSourceStatus =
  | "pending"
  | "processing"
  | "ready"
  | "failed";

export interface KnowledgeSource {
  source_id: string;
  tenant_id: string;
  original_filename: string;
  stored_filename: string;
  file_type: string;
  content_hash: string;
  status: KnowledgeSourceStatus;
  error_message?: string | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}
