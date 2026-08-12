"""
PaddleOCR-VL 图片解析模块。

负责：
加载本地图片识别模型
→ 解析单张图片
→ 保存结构化 JSON 和 Markdown
→ 批量解析第 7～11 张图的诊断区域
→ 按原始参考图合并识别结果

暂时不负责：
生成 LangChain Document
写入 Chroma
接收和对比用户上传的截图
"""


import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from paddleocr import PaddleOCRVL

from back.rag.image_text_cleaner import (
    deduplicate_region_texts,
)


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# 不再硬编码某台电脑上的 snapshot 哈希目录。
# Hugging Face 会根据模型 ID，在当前用户的缓存中找到
# 对应的 snapshots/<commit_hash> 实际目录。
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"

# 当前先关闭版面检测，只验证：
# 图片 → PaddleOCR-VL 主模型 → JSON/Markdown。
# 等单张图片识别结果确认后，再决定是否开启完整布局分析。
USE_LAYOUT_DETECTION = False

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ocr_preview"
)

# 批量解析第 6～11 张参考图时使用独立目录，避免和单图测试
# 的结果混在一起。
REFERENCE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ocr_reference"
)


@lru_cache(maxsize=1)
def get_model_path() -> Path:
    """
    从 Hugging Face 本地缓存定位模型 snapshot。

    local_files_only=True 表示：
    - 只使用已经下载到本机的模型；
    - 不在程序启动时偷偷重新下载；
    - 模型不存在时立即给出错误。

    snapshot_download 返回的不是缓存根目录，而是包含
    config.json 和 model.safetensors 的实际 snapshot 目录。
    """

    try:
        snapshot_path = snapshot_download(
            repo_id=MODEL_ID,
            local_files_only=True,
        )
    except Exception as error:
        # 保留原始异常作为 __cause__，方便排查真实原因；
        # 同时给初学者一个更明确的本地模型提示。
        raise FileNotFoundError(
            "没有在 Hugging Face 本地缓存中找到模型："
            f"{MODEL_ID}。请先下载模型。"
        ) from error

    model_path = Path(snapshot_path).resolve()

    # snapshot_download 正常情况下会保证目录完整。
    # 这里额外检查核心文件，避免缓存不完整时等到模型加载
    # 很久以后才出现难以理解的错误。
    required_files = {
        "config.json",
        "model.safetensors",
    }
    missing_files = [
        file_name
        for file_name in required_files
        if not (model_path / file_name).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "PaddleOCR-VL 模型缓存不完整，缺少："
            f"{', '.join(sorted(missing_files))}。"
            f"模型目录：{model_path}"
        )

    return model_path


@lru_cache(maxsize=1)
def get_paddle_device() -> str:
    """
    自动选择 Paddle 推理设备。

    当前开发电脑有可用 NVIDIA GPU 时返回 gpu:0；
    换到没有显卡的电脑时自动返回 cpu，代码无需再修改。
    """

    # 不要在文件顶部先导入 paddle。
    #
    # PaddleOCR/PaddleX 的导入过程会经过：
    # langchain_text_splitters → sentence_transformers → torch。
    # 在 Windows 中，如果 Paddle GPU 的 DLL 比 CPU Torch 更早
    # 加载，Torch 可能无法加载 shm.dll，并抛出 WinError 127。
    # 这里延迟导入，确保 PaddleOCR 的依赖链先完成初始化。
    import paddle

    if (
        paddle.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    ):
        return "gpu:0"

    return "cpu"


@lru_cache(maxsize=1)
def get_image_parser():
    """加载并缓存 PaddleOCR-VL 模型。"""

    model_path = get_model_path()
    device = get_paddle_device()

    print(f"PaddleOCR-VL 模型：{MODEL_ID}")
    print(f"模型实际路径：{model_path}")
    print(f"推理设备：{device}")
    print(
        "版面检测："
        f"{'开启' if USE_LAYOUT_DETECTION else '关闭'}"
    )

    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_model_dir=str(model_path),
        device=device,
        use_layout_detection=USE_LAYOUT_DETECTION,
    )

    return pipeline


def parse_image(
    image_path: str | Path,
    output_directory: str | Path = OUTPUT_DIR,
    print_result: bool = True,
    max_new_tokens: int | None = None,
):
    """
    解析一张图片并保存原始识别结果。

    当前保留 PaddleOCR-VL 的完整结果对象，下一阶段会先
    观察 JSON 结构，再决定从哪个字段提取 OCR 正文。
    """

    input_path = Path(image_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"没有找到图片：{input_path}"
        )

    output_directory = Path(
        output_directory
    ).resolve()
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = get_image_parser()

    predict_kwargs = {
        "input": str(input_path),
    }

    if max_new_tokens is not None:
        predict_kwargs["max_new_tokens"] = max_new_tokens

    results = list(
        pipeline.predict(**predict_kwargs)
    )

    if not results:
        raise RuntimeError(
            f"图片没有返回识别结果：{input_path}"
        )

    for result_index, result in enumerate(
        results,
        start=1,
    ):
        if print_result:
            result.print()

        # 使用固定文件名，批量处理时每个区域都有自己的目录，
        # 因此不会发生不同图片结果互相覆盖。
        json_path = (
            output_directory
            / f"raw_result_{result_index:02d}.json"
        )
        markdown_path = (
            output_directory
            / f"raw_result_{result_index:02d}.md"
        )

        result.save_to_json(
            save_path=str(json_path),
        )

        result.save_to_markdown(
            save_path=str(markdown_path),
        )

    return results


def _load_manifest(
    manifest_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """读取并验证 image_splitter.py 生成的清单。"""

    resolved_path = Path(manifest_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"没有找到图片分割清单：{resolved_path}"
        )

    try:
        manifest = json.loads(
            resolved_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"图片分割清单不是有效 JSON：{resolved_path}"
        ) from error

    reference_images = manifest.get(
        "reference_images"
    )

    if not isinstance(reference_images, list):
        raise ValueError(
            "图片分割清单缺少 reference_images 列表"
        )

    return resolved_path, manifest


def _calculate_file_sha256(file_path: Path) -> str:
    """计算裁剪图片哈希，用于判断缓存结果是否仍然有效。"""

    digest = sha256()

    with file_path.open("rb") as file:
        while data := file.read(1024 * 1024):
            digest.update(data)

    return digest.hexdigest()


def _write_json(
    output_path: Path,
    data: dict[str, Any],
) -> None:
    """先写临时文件再替换，避免中途中断留下半个 JSON。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _extract_result_text(
    result_payloads: list[dict[str, Any]],
) -> str:
    """按模型返回顺序合并所有文字块。"""

    text_blocks = []

    for payload in result_payloads:
        parsing_results = payload.get(
            "parsing_res_list",
            [],
        )

        for block in parsing_results:
            content = str(
                block.get("block_content", "")
            ).strip()

            if content:
                text_blocks.append(content)

    return "\n".join(text_blocks)


def _serialize_results(
    results,
) -> tuple[
    list[dict[str, Any]],
    str,
    str,
]:
    """把 Paddle 结果对象转换成可长期保存的普通数据。"""

    result_payloads = []
    markdown_parts = []

    for result in results:
        # PaddleX 的 json 属性格式是 {"res": {...}}。
        result_payloads.append(
            result.json["res"]
        )

        markdown_data = result.markdown
        markdown_text = str(
            markdown_data.get(
                "markdown_texts",
                "",
            )
        ).strip()

        if markdown_text:
            markdown_parts.append(markdown_text)

    return (
        result_payloads,
        _extract_result_text(result_payloads),
        "\n\n".join(markdown_parts),
    )


def extract_ocr_text(results) -> str:
    """从 Paddle 结果对象中提取按阅读顺序合并的纯文字。"""

    _, ocr_text, _ = _serialize_results(results)

    return ocr_text


def _release_region_result_memory(results) -> None:
    """
    一块区域完成后释放临时推理结果。

    模型本身由 get_image_parser() 缓存，不会重复加载；这里只
    清理上一块图片产生的临时张量，降低 8GB 显卡连续处理
    17 个区域时显存不足的概率。
    """

    results.clear()

    try:
        import paddle

        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        # 清理失败不应覆盖已经成功保存的 OCR 结果。
        pass


def release_result_memory(results) -> None:
    """供批量参考图和用户临时截图共用的显存清理入口。"""

    _release_region_result_memory(results)


def _parse_manifest_region(
    image_data: dict[str, Any],
    region_data: dict[str, Any],
    region_output_directory: Path,
    force: bool,
) -> dict[str, Any]:
    """识别一个区域；已有有效缓存时直接复用。"""

    crop_path = Path(
        region_data["crop_path"]
    ).resolve()

    if not crop_path.exists():
        raise FileNotFoundError(
            f"没有找到裁剪图片：{crop_path}"
        )

    crop_sha256 = _calculate_file_sha256(crop_path)
    region_result_path = (
        region_output_directory
        / "region_result.json"
    )

    # 17 个区域连续运行耗时较长。每完成一个区域就保存缓存，
    # 下次运行时可以从未完成的位置继续，而不必全部重跑。
    if region_result_path.exists() and not force:
        cached_result = json.loads(
            region_result_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            cached_result.get("status") == "completed"
            and cached_result.get("crop_sha256")
            == crop_sha256
        ):
            print("  已有有效结果，跳过模型推理")
            return cached_result

    results = parse_image(
        crop_path,
        output_directory=region_output_directory,
        print_result=False,
    )

    try:
        (
            raw_results,
            ocr_text,
            markdown_text,
        ) = _serialize_results(results)

        region_result = {
            "status": "completed",
            "image_order": image_data["image_order"],
            "category": image_data["category"],
            "region_name": region_data["region_name"],
            "region_title": region_data["region_title"],
            "role": region_data["role"],
            "description": region_data["description"],
            "crop_path": str(crop_path),
            "crop_sha256": crop_sha256,
            "normalized_box": region_data[
                "normalized_box"
            ],
            "pixel_box": region_data["pixel_box"],
            "ocr_text": ocr_text,
            "markdown_text": markdown_text,
            "raw_results": raw_results,
        }

        _write_json(
            region_result_path,
            region_result,
        )

        return region_result
    finally:
        _release_region_result_memory(results)


def _build_image_markdown(
    image_result: dict[str, Any],
) -> str:
    """把一张参考图的所有区域组织成便于人工检查的 Markdown。"""

    lines = [
        f"# 图片 {image_result['image_order']}："
        f"{image_result['title']}",
        "",
        f"- 故障类别：`{image_result['category']}`",
        f"- 所属章节：{image_result.get('section_id', '')} "
        f"{image_result.get('section_title', '')}",
        f"- 诊断目标：{image_result['diagnostic_goal']}",
        f"- 完整原图：{image_result['full_image_path']}",
        "",
    ]

    for region in image_result["regions"]:
        lines.extend(
            [
                f"## {region['region_title']}",
                "",
                f"区域作用：{region['role']}",
                "",
                region["ocr_text"] or "[没有识别到文字]",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def parse_reference_manifest(
    manifest_path: str | Path,
    output_root: str | Path = REFERENCE_OUTPUT_ROOT,
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    批量识别分割清单中的所有参考图区域。

    处理顺序严格按照 manifest.json：
    图片顺序 → 区域顺序。模型只加载一次，每个区域完成后
    立即保存，最后再合并成一张参考图的诊断结果。
    """

    resolved_manifest_path, manifest = _load_manifest(
        manifest_path
    )
    output_root = Path(output_root).resolve()
    output_directory = (
        output_root / resolved_manifest_path.parent.name
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_images = manifest["reference_images"]
    total_regions = sum(
        len(image_data.get("regions", []))
        for image_data in reference_images
    )
    completed_region_count = 0
    image_results = []

    print(f"需要识别 {total_regions} 个诊断区域")

    for image_data in reference_images:
        image_order = int(image_data["image_order"])
        image_output_directory = (
            output_directory
            / f"image_{image_order:03d}"
        )
        image_regions = []

        for region_data in image_data.get(
            "regions",
            [],
        ):
            completed_region_count += 1
            region_name = region_data["region_name"]

            print(
                f"\n[{completed_region_count}/{total_regions}] "
                f"图片 {image_order} / {region_name}"
            )

            region_output_directory = (
                image_output_directory / region_name
            )
            region_output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            region_result = _parse_manifest_region(
                image_data,
                region_data,
                region_output_directory,
                force=force,
            )
            image_regions.append(region_result)

        combined_ocr_text = deduplicate_region_texts(
            image_regions
        )

        image_result = {
            "status": "completed",
            "image_order": image_order,
            "category": image_data["category"],
            "title": image_data["title"],
            "diagnostic_goal": image_data[
                "diagnostic_goal"
            ],
            "full_image_path": image_data[
                "full_image_path"
            ],
            "section_id": image_data.get(
                "section_id",
                "",
            ),
            "section_title": image_data.get(
                "section_title",
                "",
            ),
            "block_index": image_data.get(
                "block_index",
                0,
            ),
            "previous_text": image_data.get(
                "previous_text",
                "",
            ),
            "next_text": image_data.get(
                "next_text",
                "",
            ),
            "combined_ocr_text": combined_ocr_text,
            "regions": image_regions,
        }

        image_result_path = (
            image_output_directory
            / "diagnostic_result.json"
        )
        image_markdown_path = (
            image_output_directory
            / "diagnostic_result.md"
        )

        _write_json(
            image_result_path,
            image_result,
        )
        image_markdown_path.write_text(
            _build_image_markdown(image_result),
            encoding="utf-8",
        )
        image_results.append(image_result)

    index_data = {
        "status": "completed",
        "source_manifest": str(
            resolved_manifest_path
        ),
        "image_count": len(image_results),
        "region_count": total_regions,
        "images": [
            {
                "image_order": result["image_order"],
                "category": result["category"],
                "title": result["title"],
                "result_path": str(
                    (
                        output_directory
                        / f"image_{result['image_order']:03d}"
                        / "diagnostic_result.json"
                    ).resolve()
                ),
            }
            for result in image_results
        ],
    }
    _write_json(
        output_directory / "reference_ocr_index.json",
        index_data,
    )

    return image_results


def main():
    """
    通过命令行测试单张图片。

    示例：
    python -m back.rag.image_parser <图片完整路径>
    """

    argument_parser = argparse.ArgumentParser(
        description=(
            "单图测试或批量解析故障诊断参考图"
        )
    )

    argument_parser.add_argument(
        "image_path",
        nargs="?",
        help="单张测试图片路径",
    )
    argument_parser.add_argument(
        "--manifest",
        help=(
            "image_splitter.py 生成的 manifest.json 路径"
        ),
    )
    argument_parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有区域结果并重新识别",
    )

    arguments = argument_parser.parse_args()

    if arguments.image_path and arguments.manifest:
        argument_parser.error(
            "单张图片路径和 --manifest 不能同时使用"
        )

    if not arguments.image_path and not arguments.manifest:
        argument_parser.error(
            "请提供单张图片路径，或者使用 --manifest"
        )

    if arguments.manifest:
        image_results = parse_reference_manifest(
            arguments.manifest,
            force=arguments.force,
        )

        print(
            "\n参考图片解析完成，共生成 "
            f"{len(image_results)} 份诊断结果"
        )
        print(f"输出根目录：{REFERENCE_OUTPUT_ROOT}")
        return

    results = parse_image(arguments.image_path)

    print(
        f"\n图片解析完成，共得到 "
        f"{len(results)} 个结果"
    )
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
