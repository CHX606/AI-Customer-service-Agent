"""PaddleOCR-VL 调用参数测试。"""


from back.rag import image_parser


class _FakeResult:
    def save_to_json(self, save_path):
        pass

    def save_to_markdown(self, save_path):
        pass


class _FakePipeline:
    def __init__(self):
        self.predict_kwargs = None

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [_FakeResult()]


def test_parse_image_passes_max_new_tokens(
    monkeypatch,
    tmp_path,
):
    """用户截图设置的生成上限必须传给 PaddleOCR-VL。"""

    image_path = tmp_path / "screenshot.png"
    image_path.write_bytes(b"test")
    pipeline = _FakePipeline()
    monkeypatch.setattr(
        image_parser,
        "get_image_parser",
        lambda: pipeline,
    )

    image_parser.parse_image(
        image_path,
        output_directory=tmp_path / "output",
        print_result=False,
        max_new_tokens=512,
    )

    assert pipeline.predict_kwargs == {
        "input": str(image_path.resolve()),
        "max_new_tokens": 512,
    }


def test_parse_image_keeps_default_for_reference_ocr(
    monkeypatch,
    tmp_path,
):
    """参考图批处理未指定上限时保持原来的模型默认值。"""

    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"test")
    pipeline = _FakePipeline()
    monkeypatch.setattr(
        image_parser,
        "get_image_parser",
        lambda: pipeline,
    )

    image_parser.parse_image(
        image_path,
        output_directory=tmp_path / "output",
        print_result=False,
    )

    assert pipeline.predict_kwargs == {
        "input": str(image_path.resolve()),
    }
