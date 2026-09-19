"""Quantize an already exported model without retaining a live FP32 session.

Uses the same Optimum AVX2 dynamic INT8 configuration as the package exporter.
Execution lifetime, cached shape inference and serialization differ. A staged
model is structurally checked before the final filename becomes visible.
"""
import gc
import logging
from pathlib import Path
import tempfile


def main():
    import onnx
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from onnxruntime.quantization.quant_utils import add_infer_metadata
    from transformers import AutoConfig

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("optimum.onnxruntime.quantization").setLevel(logging.INFO)
    root = Path("/app/data/models/bge-reranker-v2-m3-onnx")
    source = root / "onnx" / "model.onnx"
    target = root / "onnx" / "model_qint8_avx2.onnx"
    assert source.is_file(), "Export the original FP32 ONNX model first"
    assert not target.exists(), "Refusing to overwrite an existing INT8 model"
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    # The ONNX checker rejects symlinks and multiply linked weight files.
    # Keep the temporary graph next to the original regular weight file, so
    # neither copying weights nor weakening those checks is necessary.
    with tempfile.TemporaryDirectory(prefix="quantization-", dir=root) as temporary, \
            tempfile.NamedTemporaryFile(prefix="preinferred-", suffix=".onnx", dir=source.parent) as inferred_file:
        stage = Path(temporary)
        prepared_path = Path(inferred_file.name)
        # Run the same real shape inference before loading the weight bytes.
        # ORT's BaseQuantizer can then reuse it instead of deep-copying and
        # retaining an additional complete 2.2 GiB model during construction.
        original = onnx.load(str(source), load_external_data=False)
        for tensor in original.graph.initializer:
            for entry in tensor.external_data:
                if entry.key == "location":
                    assert Path(entry.value).name == entry.value
                    assert (source.parent / entry.value).is_file()
        print("Computing shape information from the external-data graph", flush=True)
        onnx.shape_inference.infer_shapes_path(str(source), str(prepared_path))
        prepared = onnx.load(str(prepared_path), load_external_data=False)
        assert list(original.graph.initializer) == list(prepared.graph.initializer), "Shape inference changed weight metadata"
        assert list(original.graph.node) == list(prepared.graph.node), "Shape inference changed computation nodes"
        add_infer_metadata(prepared)
        onnx.save_model(prepared, str(prepared_path))
        onnx.checker.check_model(str(prepared_path))
        del original, prepared
        gc.collect()
        print("PASS: shape inference cached; computation nodes and weight references unchanged", flush=True)
        quantizer = ORTQuantizer(prepared_path, config=config)
        print("Quantizing saved FP32 ONNX in an independent process", flush=True)
        quantizer.quantize(
            AutoQuantizationConfig.avx2(is_static=False),
            save_dir=stage,
            file_suffix="qint8_avx2",
            use_external_data_format=True,
        )
        del quantizer
        gc.collect()
        staged_model = stage / f"{prepared_path.stem}_qint8_avx2.onnx"
        onnx.checker.check_model(str(staged_model))
        metadata = onnx.load(str(staged_model), load_external_data=False)
        external_files = {
            entry.value
            for tensor in metadata.graph.initializer
            for entry in tensor.external_data
            if entry.key == "location"
        }
        for name in external_files:
            assert Path(name).name == name, "Unexpected external data location"
            destination = target.parent / name
            assert not destination.exists(), "Refusing to overwrite existing data"
            (stage / name).rename(destination)
        settings = stage / "ort_config.json"
        if settings.exists():
            destination = target.parent / "ort_config_qint8_avx2.json"
            assert not destination.exists()
            settings.rename(destination)
        staged_model.rename(target)
        print("PASS: INT8 model exported and structurally checked; inference check is next", flush=True)


if __name__ == "__main__":
    main()
