"""图片归一化、视觉编码及用于候选匹配的指纹。"""
from base64 import b64encode
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from PIL import Image, ImageOps, UnidentifiedImageError
from back.knowledge.images.semantic_models import ImageFingerprint

MAX_VISION_EDGE = 1800

def _open_normalized_image(image_source: bytes | str | Path) -> Image.Image:
    try:
        if isinstance(image_source, bytes):
            source = Image.open(BytesIO(image_source))
        else:
            source = Image.open(Path(image_source).resolve())
        with source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("无法读取用于语义理解的图片") from error

def _calculate_dhash(image: Image.Image, size: int = 8) -> str:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    grayscale = image.convert("L").resize((size + 1, size), resampling)
    try:
        get_pixels = getattr(grayscale, "get_flattened_data", grayscale.getdata)
        pixels = list(get_pixels())
    finally:
        grayscale.close()

    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            value <<= 1
            if pixels[offset + column] > pixels[offset + column + 1]:
                value |= 1
    return f"{value:0{size * size // 4}x}"

def fingerprint_image(image_source: bytes | str | Path) -> ImageFingerprint:
    """同时生成文件级 SHA256 和能容忍重新编码的 64 位 dHash。"""

    if isinstance(image_source, bytes):
        raw_bytes = image_source
    else:
        raw_bytes = Path(image_source).resolve().read_bytes()

    image = _open_normalized_image(raw_bytes)
    try:
        width, height = image.size
        dhash = _calculate_dhash(image, 8)
        detail_dhash = _calculate_dhash(image, 32)
    finally:
        image.close()
    return ImageFingerprint(
        sha256(raw_bytes).hexdigest(),
        dhash,
        detail_dhash,
        width,
        height,
    )

def _image_data_url(image_source: bytes | str | Path) -> str:
    image = _open_normalized_image(image_source)
    try:
        image.thumbnail((MAX_VISION_EDGE, MAX_VISION_EDGE))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True)
    finally:
        image.close()
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"

def dhash_distance(left: str, right: str) -> int:
    if not re.fullmatch(r"[0-9a-fA-F]+", left or ""):
        raise ValueError("左侧 dHash 格式无效")
    if not re.fullmatch(r"[0-9a-fA-F]+", right or ""):
        raise ValueError("右侧 dHash 格式无效")
    if len(left) != len(right):
        raise ValueError("两侧 dHash 长度不一致")
    return (int(left, 16) ^ int(right, 16)).bit_count()
