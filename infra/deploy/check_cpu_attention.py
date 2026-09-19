"""Compare real Paddle CPU outputs against the unmodified attention math."""
import os
from types import SimpleNamespace


def main():
    # Keep the original dependency import order used by the application.
    import torch  # noqa: F401
    import paddle
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl import _siglip
    from back.knowledge.images.cpu_attention import sliced_attention, install_cpu_attention_slicing

    paddle.seed(20260919)
    original = _siglip.eager_attention_forward
    model = SimpleNamespace(training=False)
    query = paddle.randn([2, 3, 17, 8])
    key = paddle.randn([2, 3, 19, 8])
    value = paddle.randn([2, 3, 19, 8])
    masks = [None, paddle.zeros([2, 1, 17, 19]), paddle.randn([1, 1, 1, 19]),
             paddle.randn([2, 3, 17, 19]) * 0.1]
    cases = 0
    maximum_error = 0.0
    with paddle.no_grad():
        for chunk in (1, 4, 8, 16, 32):
            for mask in masks:
                expected, _ = original(model, query, key, value, mask, scaling=8 ** -0.5)
                actual, _ = sliced_attention(original, chunk)(model, query, key, value, mask, scaling=8 ** -0.5)
                assert tuple(actual.shape) == tuple(expected.shape)
                assert bool(paddle.allclose(actual, expected, rtol=1e-5, atol=1e-6).item())
                maximum_error = max(maximum_error, float(paddle.max(paddle.abs(actual - expected)).item()))
                cases += 1
        # A full real layer includes Q/K/V projections, reshaping and output projection.
        config = _siglip.PaddleOCRVisionConfig(hidden_size=24, num_attention_heads=3, intermediate_size=48)
        layer = _siglip.SiglipAttention(config)
        layer.eval()
        hidden = paddle.randn([2, 17, 24])
        expected, _ = layer(hidden)
        os.environ["OCR_CPU_ATTENTION_CHUNK_SIZE"] = "4"
        assert install_cpu_attention_slicing()
        assert install_cpu_attention_slicing()
        actual, _ = layer(hidden)
        assert bool(paddle.allclose(actual, expected, rtol=1e-5, atol=1e-6).item())
        # Training calls must preserve the original implementation and weights.
        trained, weights = _siglip.eager_attention_forward(SimpleNamespace(training=True), query, key, value, None, scaling=8 ** -0.5)
        reference, expected_weights = original(SimpleNamespace(training=True), query, key, value, None, scaling=8 ** -0.5)
        assert weights is not None and bool(paddle.allclose(weights, expected_weights).item())
        assert bool(paddle.allclose(trained, reference).item())
    print(f"PASS: {cases} real CPU attention comparisons, projected-layer equivalence and training fallback; max_abs_error={maximum_error:.9g}", flush=True)


if __name__ == "__main__":
    main()
