"""
知识库参考图片分割模块。

当前知识库共有 11 张图片：
- 第 1～6 张不参与用户截图诊断，直接跳过；
- 第 7～11 张是故障定位参考图，需要保留完整原图，
  同时裁剪出更容易进行 OCR 和图片对比的诊断区域。

这里采用“原图 + 诊断区域”的双层结构：

原图
    保证任何界面信息都不会因为裁剪而丢失，后续可用于
    整体视觉比对和人工复查。

诊断区域
    放大真正影响故障判断的界面部分，避免 PaddleOCR-VL
    在整张复杂截图中只识别到少量显眼文字。

本模块只负责分类、裁剪和保存清单，不调用 OCR 模型，
也不写入 Chroma。运行和检查裁剪结果后，再接入
image_parser.py。
"""


import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from back.rag.loader import DOCUMENT_PATH
from PIL import Image

from back.rag.docx_image_extractor import (
    extract_docx_images,
)
from back.rag.docx_models import (
    ExtractedImageRecord,
)


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# 分割结果和清单统一放在 data/image_regions。
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "image_regions"

# 用户已确认：前六张图片不需要参与图片诊断。
SKIPPED_IMAGE_ORDERS = frozenset(range(1, 7))


@dataclass(frozen=True)
class RegionTemplate:
    """一块需要从参考图中裁剪出来的诊断区域。"""

    region_name: str
    title: str

    # 使用 0～1 的相对坐标，而不是固定像素：
    # (left, top, right, bottom)。
    # 这样即使以后替换成分辨率不同但版面相同的截图，
    # 模板仍然可以复用。
    normalized_box: tuple[float, float, float, float]

    # primary 是决定故障类型的核心区域；
    # context 是辅助理解界面的上下文区域。
    role: str
    description: str


@dataclass(frozen=True)
class ReferenceImageTemplate:
    """一张参考图对应的故障类别和分割方案。"""

    image_order: int
    category: str
    title: str
    diagnostic_goal: str
    regions: tuple[RegionTemplate, ...]


@dataclass(frozen=True)
class CroppedRegionRecord:
    """一个已经裁剪并保存的诊断区域。"""

    image_order: int
    category: str
    template_title: str
    diagnostic_goal: str
    region_name: str
    region_title: str
    role: str
    description: str
    source_image_path: str
    crop_path: str
    normalized_box: tuple[float, float, float, float]
    pixel_box: tuple[int, int, int, int]
    source_width: int
    source_height: int

    def to_dict(self) -> dict:
        """转换成可以写入 JSON 的普通字典。"""

        return asdict(self)


# 第 7～11 张图片的人工分类和诊断区域。
#
# 每张图片的原图都不会被修改或删除；下面只定义需要额外
# 生成的局部图片。区域之间有少量重叠，防止边界文字被切断。
REFERENCE_IMAGE_TEMPLATES = {
    7: ReferenceImageTemplate(
        image_order=7,
        category="shadowrocket_subscription_error",
        title="Shadowrocket 无法获取订阅节点",
        diagnostic_goal=(
            "识别 Shadowrocket 当前连接状态、报错提示和订阅节点状态。"
        ),
        regions=(
            RegionTemplate(
                region_name="connection_status",
                title="当前节点与连接状态",
                normalized_box=(0.00, 0.00, 1.00, 0.43),
                role="context",
                description=(
                    "包含当前节点、全局路由和连通性测试入口。"
                ),
            ),
            RegionTemplate(
                region_name="error_dialog",
                title="无法获取订阅节点提示",
                normalized_box=(0.20, 0.36, 0.82, 0.56),
                role="primary",
                description=(
                    "用于判断用户是否出现“不能获取订阅节点”错误。"
                ),
            ),
            RegionTemplate(
                region_name="subscription_details",
                title="订阅节点与剩余信息",
                normalized_box=(0.00, 0.43, 1.00, 0.93),
                role="primary",
                description=(
                    "包含剩余流量、重置时间、到期时间和订阅提示。"
                ),
            ),
        ),
    ),
    8: ReferenceImageTemplate(
        image_order=8,
        category="backup_subscription_address",
        title="公告中的备用订阅地址",
        diagnostic_goal=(
            "确认用户使用的是哪一种订阅地址，以及是否需要切换备用地址。"
        ),
        regions=(
            RegionTemplate(
                region_name="announcement_context",
                title="公告上半部分",
                normalized_box=(0.00, 0.00, 0.60, 0.50),
                role="context",
                description=(
                    "保留公告来源、时间和前置说明，避免只截取链接后失去语境。"
                ),
            ),
            RegionTemplate(
                region_name="highlighted_backup_address",
                title="箭头标注的备用地址",
                normalized_box=(0.03, 0.45, 0.57, 0.63),
                role="primary",
                description=(
                    "包含箭头指向的备用订阅地址及对应说明。"
                ),
            ),
            RegionTemplate(
                region_name="other_available_addresses",
                title="其他可用地址",
                normalized_box=(0.00, 0.57, 0.60, 1.00),
                role="primary",
                description=(
                    "保留公告中后续列出的其他地址，供故障切换时参考。"
                ),
            ),
        ),
    ),
    9: ReferenceImageTemplate(
        image_order=9,
        category="clash_subscription_import_error",
        title="Clash Verge 导入订阅失败",
        diagnostic_goal=(
            "识别 Clash Verge 导入订阅时出现的网络请求或 TLS 错误。"
        ),
        regions=(
            RegionTemplate(
                region_name="clash_subscription_page",
                title="Clash Verge 订阅页面",
                normalized_box=(0.22, 0.03, 0.78, 1.00),
                role="context",
                description=(
                    "保留客户端名称、当前页面和订阅输入区域。"
                ),
            ),
            RegionTemplate(
                region_name="subscription_input",
                title="订阅地址输入区域",
                normalized_box=(0.34, 0.15, 0.76, 0.34),
                role="primary",
                description=(
                    "用于确认用户正在执行订阅导入操作及输入区域状态。"
                ),
            ),
            RegionTemplate(
                region_name="import_error_message",
                title="导入失败错误提示",
                normalized_box=(0.52, 0.15, 0.76, 0.36),
                role="primary",
                description=(
                    "保留 failed to fetch、TLS verifier 等完整错误文字。"
                ),
            ),
        ),
    ),
    10: ReferenceImageTemplate(
        image_order=10,
        category="windows_manual_proxy",
        title="Windows 手动代理配置",
        diagnostic_goal=(
            "检查 Windows 系统代理是否开启，以及地址和端口是否正确。"
        ),
        regions=(
            RegionTemplate(
                region_name="proxy_settings_page",
                title="Windows 代理设置页面",
                normalized_box=(0.20, 0.00, 0.70, 1.00),
                role="context",
                description=(
                    "保留自动代理和手动代理两部分的整体设置状态。"
                ),
            ),
            RegionTemplate(
                region_name="manual_proxy_configuration",
                title="手动代理配置",
                normalized_box=(0.21, 0.43, 0.66, 0.90),
                role="primary",
                description=(
                    "包含代理开关、地址、端口、例外地址和保存按钮。"
                ),
            ),
            RegionTemplate(
                region_name="proxy_address_and_port",
                title="代理地址和端口",
                normalized_box=(0.21, 0.57, 0.43, 0.70),
                role="primary",
                description=(
                    "重点读取 127.0.0.1 和 10808 等关键配置值。"
                ),
            ),
        ),
    ),
    11: ReferenceImageTemplate(
        image_order=11,
        category="mac_app_store_account",
        title="Mac App Store 账户退出入口",
        diagnostic_goal=(
            "确认用户是否进入 App Store 商店菜单，以及能否找到退出登录入口。"
        ),
        regions=(
            RegionTemplate(
                region_name="store_menu",
                title="App Store 商店菜单",
                normalized_box=(0.38, 0.05, 0.88, 0.88),
                role="context",
                description=(
                    "保留完整商店菜单，确认菜单入口和当前界面一致。"
                ),
            ),
            RegionTemplate(
                region_name="account_actions",
                title="账户与退出登录",
                normalized_box=(0.39, 0.69, 0.87, 0.87),
                role="primary",
                description=(
                    "重点保留账户和退出登录两个操作项。"
                ),
            ),
        ),
    ),
}


def _normalized_to_pixel_box(
    normalized_box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """把 0～1 相对坐标转换成 PIL 使用的像素坐标。"""

    left, top, right, bottom = normalized_box

    if not (
        0 <= left < right <= 1
        and 0 <= top < bottom <= 1
    ):
        raise ValueError(
            f"无效的相对裁剪坐标：{normalized_box}"
        )

    # round 比直接 int 截断更接近模板设定的位置。
    pixel_box = (
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    )

    if (
        pixel_box[0] >= pixel_box[2]
        or pixel_box[1] >= pixel_box[3]
    ):
        raise ValueError(
            f"裁剪区域没有有效面积：{pixel_box}"
        )

    return pixel_box


def _split_one_reference_image(
    image_record: ExtractedImageRecord,
    template: ReferenceImageTemplate,
    output_directory: Path,
) -> list[CroppedRegionRecord]:
    """按照模板裁剪一张参考图片的所有诊断区域。"""

    source_path = Path(
        image_record.extracted_path
    ).resolve()

    if not source_path.exists():
        raise FileNotFoundError(
            f"没有找到待分割图片：{source_path}"
        )

    image_output_directory = (
        output_directory / source_path.stem
    )
    image_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cropped_records = []

    with Image.open(source_path) as image:
        width, height = image.size

        for region_order, region in enumerate(
            template.regions,
            start=1,
        ):
            pixel_box = _normalized_to_pixel_box(
                region.normalized_box,
                width,
                height,
            )

            cropped_image = image.crop(pixel_box)
            crop_name = (
                f"region_{region_order:02d}_"
                f"{region.region_name}.png"
            )
            crop_path = image_output_directory / crop_name

            # 所有裁剪结果统一保存为 PNG，避免 JPEG 再压缩导致
            # 小字号文字进一步模糊。
            cropped_image.save(
                crop_path,
                format="PNG",
            )

            cropped_records.append(
                CroppedRegionRecord(
                    image_order=image_record.image_order,
                    category=template.category,
                    template_title=template.title,
                    diagnostic_goal=template.diagnostic_goal,
                    region_name=region.region_name,
                    region_title=region.title,
                    role=region.role,
                    description=region.description,
                    source_image_path=str(source_path),
                    crop_path=str(crop_path.resolve()),
                    normalized_box=region.normalized_box,
                    pixel_box=pixel_box,
                    source_width=width,
                    source_height=height,
                )
            )

    return cropped_records


def split_diagnostic_reference_images(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> list[CroppedRegionRecord]:
    """
    提取 Word 图片，并分割第 7～11 张诊断参考图。

    返回值只包含局部区域记录；每张图片的完整原图路径会
    同时写进 manifest.json，因此不会丢失原始信息。
    """

    image_records = extract_docx_images(
        DOCUMENT_PATH
    )
    output_root = Path(output_root).resolve()

    # 使用文档哈希区分不同版本，避免 Word 更新后继续使用
    # 旧截图的分割结果。
    document_hash = (
        image_records[0].document_sha256[:12]
        if image_records
        else "empty_document"
    )
    output_directory = (
        output_root / f"reference_{document_hash}"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cropped_records = []
    reference_images = []
    skipped_images = []

    for image_record in image_records:
        if image_record.image_order in SKIPPED_IMAGE_ORDERS:
            skipped_images.append(
                {
                    "image_order": image_record.image_order,
                    "image_path": image_record.extracted_path,
                    "reason": "用户确认第 1～6 张不参与图片诊断",
                }
            )
            continue

        template = REFERENCE_IMAGE_TEMPLATES.get(
            image_record.image_order
        )

        if template is None:
            raise ValueError(
                "需要处理的参考图片没有分割模板："
                f"image_order={image_record.image_order}"
            )

        image_crops = _split_one_reference_image(
            image_record,
            template,
            output_directory,
        )
        cropped_records.extend(image_crops)

        reference_images.append(
            {
                "image_order": image_record.image_order,
                "category": template.category,
                "title": template.title,
                "diagnostic_goal": template.diagnostic_goal,
                "full_image_path": image_record.extracted_path,
                "section_id": image_record.section_id,
                "section_title": image_record.section_title,
                "block_index": image_record.block_index,
                "previous_text": image_record.previous_text,
                "next_text": image_record.next_text,
                "regions": [
                    crop.to_dict()
                    for crop in image_crops
                ],
            }
        )

    manifest = {
        "strategy": (
            "保留完整原图，同时裁剪关键诊断区域；"
            "第 1～6 张跳过，第 7～11 张参与用户截图诊断。"
        ),
        "skipped_images": skipped_images,
        "reference_images": reference_images,
    }

    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return cropped_records


def main():
    """命令行入口：生成裁剪图片和分割清单。"""

    argument_parser = argparse.ArgumentParser(
        description="分割第 7～11 张故障诊断参考图"
    )
    argument_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="裁剪结果输出目录",
    )
    arguments = argument_parser.parse_args()

    records = split_diagnostic_reference_images(
        output_root=arguments.output_dir,
    )

    print(
        "诊断参考图分割完成，共生成 "
        f"{len(records)} 个区域。"
    )
    print(f"输出根目录：{Path(arguments.output_dir).resolve()}")


if __name__ == "__main__":
    main()
