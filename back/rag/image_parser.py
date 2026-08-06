"""
PaddleOCR-VL 图片解析模块。

负责：
加载本地图片识别模型
→ 解析单张图片
→ 保存结构化 JSON 和 Markdown

暂时不负责：
提取 Word 图片
生成 LangChain Document
写入 Chroma
"""


import argparse
from functools import lru_cache
from pathlib import Path

from paddleocr import PaddleOCRVL


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

MODEL_PATH = Path(
    r"D:\AI-models\PaddleOCR-VL-1.6"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ocr_preview"
)


@lru_cache(maxsize=1)
def get_image_parser():
    """加载并缓存 PaddleOCR-VL 模型。"""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"没有找到图片模型：{MODEL_PATH}"
        )

    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_model_dir=str(MODEL_PATH),
        device="gpu:0",
        engine="paddle",
    )

    return pipeline


def parse_image(
    image_path: str | Path,
):
    """解析一张图片并保存原始识别结果。"""

    input_path = Path(image_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"没有找到图片：{input_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = get_image_parser()

    results = list(
        pipeline.predict(
            input=str(input_path),
        )
    )

    if not results:
        raise RuntimeError(
            f"图片没有返回识别结果：{input_path}"
        )

    for result in results:
        result.print()

        result.save_to_json(
            save_path=OUTPUT_DIR,
        )

        result.save_to_markdown(
            save_path=OUTPUT_DIR,
        )

    return results


def main():
    """通过命令行测试单张图片。"""

    argument_parser = argparse.ArgumentParser(
        description="测试 PaddleOCR-VL 图片解析"
    )

    argument_parser.add_argument(
        "image_path",
        help="需要解析的图片路径",
    )

    arguments = argument_parser.parse_args()

    results = parse_image(
        arguments.image_path
    )

    print(
        f"\n图片解析完成，共得到 "
        f"{len(results)} 个结果"
    )

    print(
        f"输出目录：{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()