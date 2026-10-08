"""临时截图区域 OCR、结果释放以及完整截图到分区的降级流程。"""

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import monotonic

from PIL import Image

from back.knowledge.images.query_models import (
    MAX_SINGLE_REGION_PIXELS, USER_IMAGE_MAX_NEW_TOKENS, QueryRegion, UserImageAnalysis,
)
from back.knowledge.images.query_regions import _build_query_regions, _to_pixel_box
from back.knowledge.images.text_cleaner import deduplicate_text_lines, is_ocr_text_sufficient


def recognize_query_region(
    image: Image.Image, region: QueryRegion, *, region_index: int, region_total: int,
    temporary_path: Path, parse: Callable, extract: Callable, release: Callable,
) -> str:
    """识别一个临时区域；解析依赖由公开入口传入，保持替换能力。"""
    width, height = image.size
    region_image = image.crop(_to_pixel_box(region.box, width, height))
    region_path = temporary_path / f"region_{region_index:02d}_{region.name}.png"
    try:
        region_image.save(region_path, format="PNG")
    finally:
        region_image.close()
    print(f"识别用户截图区域：{region_index}/{region_total} {region.name}")
    started_at = monotonic()
    results = parse(
        region_path, output_directory=temporary_path / f"ocr_{region_index:02d}",
        print_result=False, max_new_tokens=USER_IMAGE_MAX_NEW_TOKENS,
    )
    try:
        return extract(results).strip()
    finally:
        release(results)
        print(
            f"完成用户截图区域：{region_index}/{region_total} {region.name}，耗时 "
            f"{monotonic() - started_at:.1f} 秒"
        )


def _recognize_full_image(image: Image.Image, temporary_path: Path, recognize: Callable):
    try:
        text = recognize(
            image, QueryRegion("full_image", (0.0, 0.0, 1.0, 1.0)),
            region_index=1, region_total=1, temporary_path=temporary_path,
        )
        return text, None
    except Exception as error:
        print(f"完整截图识别失败，将判断是否分区兜底：{error}")
        return "", error


def _recognize_fallback_regions(
    image: Image.Image, regions: tuple[QueryRegion, ...], temporary_path: Path,
    recognize: Callable,
) -> list[str]:
    texts = []
    for index, region in enumerate(regions, start=2):
        try:
            text = recognize(
                image, region, region_index=index, region_total=1 + len(regions),
                temporary_path=temporary_path,
            )
        except Exception as error:
            print(f"截图兜底区域 {region.name} 识别失败：{error}")
            continue
        if text:
            texts.append(text)
    return texts


def _run_ocr(
    image: Image.Image, temporary_path: Path, recognize: Callable,
) -> UserImageAnalysis:
    width, height = image.size
    full_text, full_error = _recognize_full_image(image, temporary_path, recognize)
    texts = [full_text] if full_text else []
    layout, count = "full_image", 1
    if width * height > MAX_SINGLE_REGION_PIXELS and not is_ocr_text_sufficient(full_text):
        fallback_layout, regions = _build_query_regions(width, height)
        layout, count = f"full_then_{fallback_layout}", 1 + len(regions)
        texts.extend(_recognize_fallback_regions(image, regions, temporary_path, recognize))
    elif full_error is not None:
        raise full_error
    ocr_text = deduplicate_text_lines(texts)
    if not ocr_text:
        raise RuntimeError("没有从用户截图中识别到可用于诊断的文字")
    return UserImageAnalysis(
        width=width, height=height, layout=layout, region_count=count, ocr_text=ocr_text,
    )


def analyze_ocr(
    image: Image.Image, recognize: Callable, inference_lock: Lock,
) -> UserImageAnalysis:
    """序列化单 GPU 推理，临时图片及 OCR 文件在退出时一并删除。"""
    with inference_lock:
        with TemporaryDirectory(prefix="customer_service_image_") as directory:
            return _run_ocr(image, Path(directory), recognize)

