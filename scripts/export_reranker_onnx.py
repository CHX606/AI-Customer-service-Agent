"""一次性导出 CrossEncoder 的 ONNX O3 与动态 INT8 版本。"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import (
    CrossEncoder,
    export_dynamic_quantized_onnx_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "bge-reranker-v2-m3-onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--variant",
        choices=["all", "o3", "int8"],
        default="all",
    )
    parser.add_argument(
        "--quantization",
        choices=["avx2", "avx512", "avx512_vnni"],
        default="avx512",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_length < 1:
        raise ValueError("max-length 必须大于 0")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"加载并导出 ONNX 基础模型：{args.model}", flush=True)
    model = CrossEncoder(
        args.model,
        backend="onnx",
        max_length=args.max_length,
        model_kwargs={
            "export": True,
            "provider": "CPUExecutionProvider",
        },
    )
    model.save_pretrained(str(output))

    if args.variant in {"all", "o3"}:
        print("生成 ONNX O3 FP32 版本…", flush=True)
        from optimum.onnxruntime import ORTOptimizer
        from optimum.onnxruntime.configuration import AutoOptimizationConfig

        optimizer = ORTOptimizer.from_pretrained(model.transformers_model)
        optimizer.optimize(
            optimization_config=AutoOptimizationConfig.O3(),
            save_dir=output / "onnx",
            file_suffix="O3",
            one_external_file=True,
        )

    if args.variant in {"all", "int8"}:
        print(f"生成 ONNX INT8 {args.quantization} 版本…", flush=True)
        export_dynamic_quantized_onnx_model(
            model=model,
            quantization_config=args.quantization,
            model_name_or_path=str(output),
            file_suffix=f"qint8_{args.quantization}",
        )

    print(f"导出目录：{output}")
    for path in sorted(output.rglob("*.onnx")):
        print(f"- {path.relative_to(output)} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
