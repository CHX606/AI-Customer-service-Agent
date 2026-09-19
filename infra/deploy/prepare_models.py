"""Prepare and load-check all local models inside the backend container."""
from pathlib import Path
import os
import subprocess
import sys


def run(*args):
    subprocess.run([sys.executable, *args], check=True)


def main():
    os.chdir(Path(__file__).resolve().parents[2])
    # Separate processes release model RAM between export/load checks.
    print("Preparing 512-dimensional Chinese embeddings", flush=True)
    run("-c", "from back.knowledge.retrieval.embeddings import get_embedding_model; "
        "v=get_embedding_model().embed_query('部署检查'); "
        "assert len(v)==512, f'Unexpected embedding dimension: {len(v)}'")
    target = Path("data/models/bge-reranker-v2-m3-onnx/onnx/model_qint8_avx2.onnx")
    if not target.is_file():
        # Release the FP32 inference session before the quantizer loads weights.
        # Reuse a completed base export after an interrupted quantization.
        if not target.with_name("model.onnx").is_file():
            run("-m", "scripts.export_reranker_onnx", "--variant", "fp32")
        run("infra/deploy/quantize_existing_reranker.py")
    print("Checking ONNX reranker inference", flush=True)
    run("-c", "import math; from back.knowledge.retrieval.reranker import get_reranker_model; "
        "scores=get_reranker_model().predict([('如何联系客服','请通过在线工单联系客服')]); "
        "assert len(scores)==1 and math.isfinite(float(scores[0]))")
    from back.core.features import local_ocr_enabled
    if not local_ocr_enabled():
        print("PASS: local retrieval models ready; local OCR explicitly disabled", flush=True)
        return
    print("Downloading and load-checking PaddleOCR-VL-1.6", flush=True)
    run("-c", "from huggingface_hub import snapshot_download; "
        "snapshot_download(repo_id='PaddlePaddle/PaddleOCR-VL-1.6')")
    run("-c", "from back.knowledge.images.parser import get_image_parser; "
        "get_image_parser(); print('OCR model load OK')")
    print("All model preparation checks passed.", flush=True)


if __name__ == "__main__":
    main()
