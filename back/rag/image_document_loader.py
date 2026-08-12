"""
把图片 OCR 诊断结果转换成 LangChain Document。

输入：
data/ocr_reference/reference_<文档哈希>/diagnostic_result.json

输出：
每张参考图对应一个 Document，其中：
- page_content 保存可参与向量检索和 BM25 检索的诊断文字；
- metadata 保存图片类别、Word 位置和原图路径；
- 不重复写入各区域的 raw_result。
"""


from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from back.rag.loader import DOCUMENT_PATH


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
OCR_REFERENCE_ROOT = PROJECT_ROOT / "data" / "ocr_reference"


def _calculate_file_sha256(file_path: Path) -> str:
    """计算当前 Word 文档哈希，用于找到同版本 OCR 结果。"""

    digest = sha256()

    with file_path.open("rb") as file:
        while data := file.read(1024 * 1024):
            digest.update(data)

    return digest.hexdigest()


def get_reference_ocr_index_path(
    document_path: str | Path = DOCUMENT_PATH,
) -> Path:
    """定位与当前 Word 版本完全对应的图片 OCR 索引。"""

    resolved_document_path = Path(
        document_path
    ).resolve()

    if not resolved_document_path.exists():
        raise FileNotFoundError(
            f"没有找到 Word 知识库：{resolved_document_path}"
        )

    document_hash = _calculate_file_sha256(
        resolved_document_path
    )

    return (
        OCR_REFERENCE_ROOT
        / f"reference_{document_hash[:12]}"
        / "reference_ocr_index.json"
    )


def _read_json(json_path: Path) -> dict[str, Any]:
    """读取 JSON，并把路径和格式错误转换成明确提示。"""

    if not json_path.exists():
        raise FileNotFoundError(
            f"没有找到图片诊断结果：{json_path}"
        )

    try:
        data = json.loads(
            json_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"图片诊断结果不是有效 JSON：{json_path}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"图片诊断结果必须是 JSON 对象：{json_path}"
        )

    return data


def _build_page_content(
    image_result: dict[str, Any],
) -> str:
    """组织真正用于 Embedding、BM25 和 Reranker 的文字。"""

    title = str(image_result.get("title", "")).strip()
    category = str(
        image_result.get("category", "")
    ).strip()
    diagnostic_goal = str(
        image_result.get("diagnostic_goal", "")
    ).strip()
    ocr_text = str(
        image_result.get("combined_ocr_text", "")
    ).strip()

    if not title:
        raise ValueError("图片诊断结果缺少 title")

    if not diagnostic_goal:
        raise ValueError(
            f"图片 {title} 缺少 diagnostic_goal"
        )

    if not ocr_text:
        raise ValueError(
            f"图片 {title} 没有可检索的 OCR 文字"
        )

    # 标题和诊断目标比原始 OCR 更稳定，放在正文前面可以帮助
    # 用户使用自然语言描述故障时召回正确参考图。
    return "\n".join(
        [
            "[图片诊断资料]",
            f"诊断名称：{title}",
            f"故障类别：{category}",
            f"诊断目标：{diagnostic_goal}",
            "图片识别信息：",
            ocr_text,
        ]
    )


def _resolve_result_path(
    index_path: Path,
    image_order: int,
) -> Path:
    """
    按索引所在目录拼接结果路径。

    不直接依赖 index 中保存的绝对路径，这样整个项目移动到
    另一台电脑后，只要 data 目录结构不变，仍然可以加载。
    """

    return (
        index_path.parent
        / f"image_{image_order:03d}"
        / "diagnostic_result.json"
    )


def load_image_documents(
    index_path: str | Path | None = None,
) -> list[Document]:
    """读取第 7～11 张图片结果并生成可直接入库的 Document。"""

    resolved_index_path = (
        Path(index_path).resolve()
        if index_path is not None
        else get_reference_ocr_index_path().resolve()
    )
    index_data = _read_json(resolved_index_path)

    if index_data.get("status") != "completed":
        raise ValueError(
            f"图片 OCR 索引尚未完成：{resolved_index_path}"
        )

    image_entries = index_data.get("images")

    if not isinstance(image_entries, list):
        raise ValueError(
            "图片 OCR 索引缺少 images 列表"
        )

    documents = []

    for image_entry in image_entries:
        image_order = int(
            image_entry["image_order"]
        )
        result_path = _resolve_result_path(
            resolved_index_path,
            image_order,
        )
        image_result = _read_json(result_path)

        if image_result.get("status") != "completed":
            raise ValueError(
                f"图片 {image_order} 的诊断结果尚未完成"
            )

        full_image_path = str(
            image_result.get("full_image_path", "")
        )

        documents.append(
            Document(
                page_content=_build_page_content(
                    image_result
                ),
                metadata={
                    # 每张图片使用不同的诊断结果文件作为 source，
                    # 确保混合检索的文档唯一键不会冲突。
                    "source": str(result_path.resolve()),
                    "document_name": DOCUMENT_PATH.name,
                    "content_type": "image_diagnostic",
                    "image_order": image_order,
                    "category": str(
                        image_result.get("category", "")
                    ),
                    "diagnostic_title": str(
                        image_result.get("title", "")
                    ),
                    "section_id": str(
                        image_result.get("section_id", "")
                    ),
                    "section_title": str(
                        image_result.get("section_title", "")
                    ),
                    "block_index": int(
                        image_result.get("block_index", 0)
                    ),
                    # 图片 Document 已经是最小完整诊断单元，
                    # 不再二次切块，因此固定为第 0 块。
                    "chunk_index": 0,
                    "full_image_path": full_image_path,
                    "ocr_result_path": str(
                        result_path.resolve()
                    ),
                },
            )
        )

    expected_count = int(
        index_data.get("image_count", len(documents))
    )

    if len(documents) != expected_count:
        raise ValueError(
            "图片 OCR 索引数量不一致："
            f"索引声明 {expected_count}，实际加载 {len(documents)}"
        )

    return documents


if __name__ == "__main__":
    image_documents = load_image_documents()

    print(
        "图片诊断 Document 加载成功，共 "
        f"{len(image_documents)} 个"
    )

    for document in image_documents:
        print("\n" + "=" * 70)
        print(document.metadata)
        print(document.page_content)
