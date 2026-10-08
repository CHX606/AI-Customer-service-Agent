import { Alert, Button, Card, Empty, Flex, Space, Spin, Tag, Tooltip, Typography, Upload } from "antd";
import { FileText, RefreshCw, RotateCcw, Trash2, Upload as UploadIcon } from "lucide-react";
import type { useKnowledgeSources } from "../hooks/use-knowledge-sources";
import type { KnowledgeSource } from "../types/profile";

type KnowledgeController = ReturnType<typeof useKnowledgeSources>;
function KnowledgeCard({ file, knowledge }: { file: KnowledgeSource; knowledge: KnowledgeController }) {
  return <Card size="small"><Flex align="start" gap={12}>
    <FileText size={22} className="file-icon" />
    <div className="file-info">
      <Typography.Text strong className="file-name">{file.original_filename}</Typography.Text>
      <Flex wrap gap={6} className="file-tags"><Tag>{file.file_type.toUpperCase()}</Tag>
        <Tag color={file.status === "ready" ? "success" : file.status === "failed" ? "error" : "processing"}>
          {{ ready: "可用于回答", failed: "处理失败", processing: "处理中", pending: "等待处理" }[file.status]}
        </Tag>
      </Flex>
      <Typography.Text type="secondary" className="file-meta">{file.chunk_count} 个内容片段 · 更新于 {new Date(file.updated_at).toLocaleString("zh-CN")}</Typography.Text>
      {file.error_message && <Typography.Paragraph type="danger">{file.error_message}</Typography.Paragraph>}
    </div>
    <Space size={4}>
      <Tooltip title="重新处理"><Button type="text" size="small" aria-label={`重新处理 ${file.original_filename}`}
        icon={<RotateCcw size={16} />} disabled={Boolean(knowledge.operatingId) || file.status === "processing"}
        onClick={() => knowledge.handleReindexFile(file.source_id)} /></Tooltip>
      <Tooltip title="删除文档"><Button type="text" size="small" danger aria-label={`删除 ${file.original_filename}`}
        icon={<Trash2 size={16} />} disabled={Boolean(knowledge.operatingId)} onClick={() => knowledge.handleDeleteFile(file.source_id)} /></Tooltip>
    </Space>
  </Flex></Card>;
}
export function AdminKnowledge({ knowledge }: { knowledge: KnowledgeController }) {
  return <Flex vertical gap={20} className="knowledge-content">
    <Upload.Dragger name="file" accept=".docx,.pdf,.txt,.md,.markdown" multiple={false} showUploadList={false}
      disabled={knowledge.isUploading} beforeUpload={file => { void knowledge.handleFileUpload(file); return false; }}>
      <Flex vertical align="center" gap={8}>
        {knowledge.isUploading ? <Spin /> : <UploadIcon size={28} />}
        <Typography.Text strong>{knowledge.isUploading ? "正在上传并处理文档…" : "点击选择文档，或将文件拖到这里"}</Typography.Text>
        <Typography.Text type="secondary">DOCX、PDF、TXT、Markdown · 单文件不超过 20MB</Typography.Text>
      </Flex>
    </Upload.Dragger>
    <Typography.Text type="secondary" className="knowledge-hint">支持文档正文；DOCX 图片会按服务器配置进行语义解析，PDF 暂仅提取文字。</Typography.Text>
    {knowledge.uploadError && <Alert type="error" showIcon title={knowledge.uploadError} />}
    <Flex justify="space-between" align="center" gap={12}>
      <Typography.Text strong>知识库文档 <Typography.Text type="secondary">({knowledge.files.length})</Typography.Text></Typography.Text>
      <Button icon={<RefreshCw size={15} />} loading={knowledge.isLoadingFiles} onClick={knowledge.loadKnowledgeData}>刷新</Button>
    </Flex>
    <Spin spinning={knowledge.isLoadingFiles}>
      {knowledge.files.length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={knowledge.uploadError ? "暂时无法加载文档，请稍后重试" : "还没有文档，上传后即可用于客服回答"} />
        : <Flex vertical gap={12}>{knowledge.files.map(file => <KnowledgeCard key={file.source_id} file={file} knowledge={knowledge} />)}</Flex>}
    </Spin>
  </Flex>;
}
