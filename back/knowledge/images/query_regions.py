"""用户截图校验和覆盖完整画面的重叠分区。"""

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from back.knowledge.images.query_models import (
    LANDSCAPE_RATIO, MAX_IMAGE_PIXELS, MAX_SINGLE_REGION_PIXELS,
    MAX_UPLOAD_BYTES, MIN_IMAGE_SIDE, PORTRAIT_RATIO,
    QueryRegion, UserImageValidationError,
)


def _load_and_validate_image(image_bytes: bytes) -> Image.Image:
    """根据真实文件内容校验图片，恢复 EXIF 方向并统一为 RGB。"""
    if not image_bytes:
        raise UserImageValidationError("上传的图片为空")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise UserImageValidationError("图片不能超过 10MB")
    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            if (source_image.format or "").upper() not in {"JPEG", "PNG", "WEBP"}:
                raise UserImageValidationError("只支持 JPG、PNG 和 WEBP 图片")
            normalized_image = ImageOps.exif_transpose(source_image).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise UserImageValidationError("上传文件不是有效图片") from error
    width, height = normalized_image.size
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        normalized_image.close()
        raise UserImageValidationError("图片尺寸太小，无法可靠识别")
    if width * height > MAX_IMAGE_PIXELS:
        normalized_image.close()
        raise UserImageValidationError("图片像素过大，请先压缩后再上传")
    return normalized_image


def _build_query_regions(width: int, height: int) -> tuple[str, tuple[QueryRegion, ...]]:
    """按照方向分区并保留重叠，避免文字或弹窗恰好跨越边界。"""
    if width * height <= MAX_SINGLE_REGION_PIXELS:
        return "single", (QueryRegion("full_image", (0.0, 0.0, 1.0, 1.0)),)
    if width / height >= LANDSCAPE_RATIO:
        return "landscape_columns", (
            QueryRegion("left", (0.00, 0.00, 0.40, 1.00)),
            QueryRegion("center", (0.30, 0.00, 0.70, 1.00)),
            QueryRegion("right", (0.60, 0.00, 1.00, 1.00)),
        )
    if width / height <= PORTRAIT_RATIO:
        return "portrait_rows", (
            QueryRegion("top", (0.00, 0.00, 1.00, 0.40)),
            QueryRegion("middle", (0.00, 0.30, 1.00, 0.70)),
            QueryRegion("bottom", (0.00, 0.60, 1.00, 1.00)),
        )
    return "square_grid", (
        QueryRegion("top_left", (0.00, 0.00, 0.60, 0.60)),
        QueryRegion("top_right", (0.40, 0.00, 1.00, 0.60)),
        QueryRegion("bottom_left", (0.00, 0.40, 0.60, 1.00)),
        QueryRegion("bottom_right", (0.40, 0.40, 1.00, 1.00)),
    )


def _to_pixel_box(
    box: tuple[float, float, float, float], width: int, height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return round(left * width), round(top * height), round(right * width), round(bottom * height)

