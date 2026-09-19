"""Opt-in, full-resolution CPU attention slicing for the pinned PaddleX build.

Only query rows are split; every row still attends to every key. No resizing,
quantization, token truncation, or training behavior is changed. SiglipAttention
discards the attention weights, so only its output is retained between chunks.
"""
import ast
from functools import wraps
import hashlib
from importlib.metadata import version
import inspect
import os
import textwrap

_EXPECTED_AST = "b72249cb3c7238e9cf59e0064106a1dea46cbce4db4504581128ac3537e54e6e"


def sliced_attention(original, chunk_size):
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Attention chunk size must be a positive integer")

    @wraps(original)
    def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        import paddle

        if (module.training or not query.place.is_cpu_place() or len(query.shape) != 4
                or query.shape[-2] <= chunk_size or kwargs.get("output_attentions", False)):
            return original(module, query, key, value, attention_mask,
                            scaling=scaling, dropout=dropout, **kwargs)
        length = query.shape[-2]
        if attention_mask is not None and attention_mask.shape[-2] not in (1, length):
            raise ValueError("Unsupported attention mask query dimension")
        outputs = []
        for start in range(0, length, chunk_size):
            stop = min(start + chunk_size, length)
            mask = attention_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., start:stop, :]
            output, weights = original(module, query[..., start:stop, :], key, value,
                                       mask, scaling=scaling, dropout=dropout, **kwargs)
            outputs.append(output)
            del weights
        return paddle.concat(outputs, axis=1), None

    forward._acs_attention_sliced = True
    return forward


def install_cpu_attention_slicing():
    chunk_size = int(os.getenv("OCR_CPU_ATTENTION_CHUNK_SIZE", "0"))
    if chunk_size == 0:
        return False
    if chunk_size < 1 or chunk_size > 8192:
        raise ValueError("OCR_CPU_ATTENTION_CHUNK_SIZE must be 0 or between 1 and 8192")
    if version("paddlex") != "3.7.2" or version("paddlepaddle") != "3.2.2":
        raise RuntimeError("CPU attention slicing requires the verified PaddleX/Paddle versions")
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl import _siglip

    original = _siglip.eager_attention_forward
    if getattr(original, "_acs_attention_sliced", False):
        return True
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fingerprint = hashlib.sha256(ast.dump(tree.body[0], include_attributes=False).encode()).hexdigest()
    if fingerprint != _EXPECTED_AST:
        raise RuntimeError("PaddleX attention implementation changed; revalidate before enabling slicing")
    _siglip.eager_attention_forward = sliced_attention(original, chunk_size)
    print(f"CPU OCR attention slicing enabled: {chunk_size} query rows; full image resolution retained", flush=True)
    return True
