import { useState } from "react";
import { App } from "antd";
import { fetchKnowledgeFiles, uploadKnowledgeFile, deleteKnowledgeFile, reindexKnowledgeFile } from "../api/profile";
import { getErrorMessage } from "../api/client";
import type { KnowledgeSource } from "../types/profile";

export function useKnowledgeSources(tenantId: string) {
  const { modal, message } = App.useApp();
  const [files, setFiles] = useState<KnowledgeSource[]>([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [operatingId, setOperatingId] = useState<string | null>(null);

  const loadKnowledgeData = async () => {
    setIsLoadingFiles(true);
    setUploadError(null);
    try { setFiles(await fetchKnowledgeFiles(tenantId)); }
    catch (error: unknown) { setUploadError(getErrorMessage(error, "加载知识库文件列表失败")); }
    finally { setIsLoadingFiles(false); }
  };
  const handleFileUpload = async (file: File) => {
    if (!/\.(docx|pdf|txt|md|markdown)$/i.test(file.name)) { message.warning("请选择 DOCX、PDF、TXT 或 Markdown 文档"); return; }
    if (file.size > 20 * 1024 * 1024) { message.warning("文件不能超过 20MB"); return; }
    setIsUploading(true);
    setUploadError(null);
    try {
      await uploadKnowledgeFile(file, tenantId);
      await loadKnowledgeData();
      message.success("文档已上传");
    } catch (error: unknown) { setUploadError(getErrorMessage(error, "文件上传失败")); }
    finally { setIsUploading(false); }
  };
  const handleDeleteFile = (sourceId: string) => {
    const file = files.find((item) => item.source_id === sourceId);
    modal.confirm({
      title: "删除知识库文档",
      content: `确定删除「${file?.original_filename || "该文档"}」吗？删除后，该文档将不再用于回答问题，且无法恢复。`,
      centered: true, okText: "确认删除", cancelText: "取消", okButtonProps: { danger: true },
      onOk: async () => {
        setOperatingId(sourceId);
        try {
          await deleteKnowledgeFile(sourceId, tenantId);
          await loadKnowledgeData();
          message.success("文档已删除");
        } catch (error: unknown) {
          message.error(getErrorMessage(error, "删除文档失败"));
          throw error;
        } finally { setOperatingId(null); }
      },
    });
  };
  const handleReindexFile = async (sourceId: string) => {
    setOperatingId(sourceId);
    try {
      await reindexKnowledgeFile(sourceId, tenantId);
      await loadKnowledgeData();
      message.success("已提交重新处理");
    } catch (error: unknown) { message.error(getErrorMessage(error, "重新处理失败")); }
    finally { setOperatingId(null); }
  };
  return { files, isLoadingFiles, isUploading, uploadError, operatingId, loadKnowledgeData, handleFileUpload, handleDeleteFile, handleReindexFile };
}
