import { requestJson } from "./client";
import type {
  KnowledgeSource,
  TenantProfile,
  TenantProfileUpdate,
  TenantPublicProfile,
} from "../types/profile";

export const DEFAULT_TENANT_ID =
  (import.meta.env.VITE_TENANT_ID as string | undefined) || "default";

function tenantQuery(tenantId: string): string {
  return `tenant_id=${encodeURIComponent(tenantId)}`;
}

export async function fetchPublicProfile(
  tenantId: string = DEFAULT_TENANT_ID,
): Promise<TenantPublicProfile> {
  return requestJson<TenantPublicProfile>(
    `/public/profile?${tenantQuery(tenantId)}`,
    {
      method: "GET",
    },
    "获取企业公开资料失败",
  );
}

export async function fetchAdminProfile(
  tenantId: string = DEFAULT_TENANT_ID,
): Promise<TenantProfile> {
  return requestJson<TenantProfile>(
    `/admin/profile?${tenantQuery(tenantId)}`,
    {
      method: "GET",
    },
    "获取后台配置失败",
  );
}

export async function updateAdminProfile(
  updateData: TenantProfileUpdate,
): Promise<TenantProfile> {
  return requestJson<TenantProfile>(
    "/admin/profile",
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(updateData),
    },
    "更新配置失败",
  );
}

export async function fetchKnowledgeFiles(
  tenantId: string = DEFAULT_TENANT_ID,
): Promise<KnowledgeSource[]> {
  return requestJson<KnowledgeSource[]>(
    `/admin/knowledge/files?${tenantQuery(tenantId)}`,
    {
      method: "GET",
    },
    "获取知识库列表失败",
  );
}

export async function uploadKnowledgeFile(
  file: File,
  tenantId: string = DEFAULT_TENANT_ID,
): Promise<KnowledgeSource> {
  const formData = new FormData();
  formData.append("file", file, file.name);
  formData.append("tenant_id", tenantId);

  return requestJson<KnowledgeSource>(
    "/admin/knowledge/files",
    {
      method: "POST",
      body: formData,
    },
    "知识库文件上传失败",
  );
}

export async function deleteKnowledgeFile(
  sourceId: string,
  tenantId: string = DEFAULT_TENANT_ID,
): Promise<void> {
  await requestJson<unknown>(
    `/admin/knowledge/files/${encodeURIComponent(sourceId)}?${tenantQuery(tenantId)}`,
    {
      method: "DELETE",
    },
    "删除知识库文件失败",
  );
}

export async function reindexKnowledgeFile(
  sourceId: string,
  tenantId: string = DEFAULT_TENANT_ID,
): Promise<KnowledgeSource> {
  return requestJson<KnowledgeSource>(
    `/admin/knowledge/files/${encodeURIComponent(sourceId)}/reindex?${tenantQuery(tenantId)}`,
    {
      method: "POST",
    },
    "重新索引失败",
  );
}
