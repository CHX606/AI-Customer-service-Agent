"""用户截图分析的输入限制、区域与结果类型。"""

from dataclasses import dataclass

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MIN_IMAGE_SIDE = 80
MAX_SINGLE_REGION_PIXELS = 600_000
LANDSCAPE_RATIO = 1.25
PORTRAIT_RATIO = 0.80
USER_IMAGE_MAX_NEW_TOKENS = 512
MIN_VISUAL_SEMANTIC_CONFIDENCE = 0.35


@dataclass(frozen=True)
class QueryRegion:
    """用户截图中的一个临时 OCR 区域。"""

    name: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class UserImageAnalysis:
    """用户图片处理后交给 Agent 的结构化结果。"""

    width: int
    height: int
    layout: str
    region_count: int
    ocr_text: str
    understanding_strategy: str = "ocr"
    semantic_text: str = ""
    matched_image_order: int | None = None
    match_distance: int | None = None


class UserImageValidationError(ValueError):
    """用户上传的文件不是系统允许处理的图片。"""

