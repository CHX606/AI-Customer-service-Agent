"""Exercise the configured API with tiny real text and vision requests.

The private configuration is mounted read-only at /run/deployment.env.
Never print credentials, response bodies or exception bodies.
"""
import base64
from io import BytesIO
import json
import time

from dotenv import load_dotenv


def main():
    load_dotenv("/run/deployment.env", override=True)
    from PIL import Image
    from langchain_core.messages import HumanMessage
    from back.core.llm import (
        get_default_model,
        get_image_understanding_model,
        get_response_model,
    )

    fixture = BytesIO()
    Image.new("RGB", (128, 128), "red").save(fixture, format="PNG")
    encoded = base64.b64encode(fixture.getvalue()).decode("ascii")
    checks = [
        ("default_text", get_default_model,
         [HumanMessage(content="Reply with exactly DEPLOY_OK and nothing else.")],
         lambda text: "DEPLOY_OK" in text),
        ("response_reasoning_configuration", get_response_model,
         [HumanMessage(content="Reply with exactly DEPLOY_OK and nothing else.")],
         lambda text: "DEPLOY_OK" in text),
        ("vision", get_image_understanding_model,
         [HumanMessage(content=[
             {"type": "text", "text": "What is the dominant color in this image? Reply with one English color word only."},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
         ])], lambda text: "red" in text.lower()),
    ]
    failures = 0
    for name, factory, messages, validate in checks:
        started = time.monotonic()
        try:
            result = factory().invoke(messages)
            content = result.content
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            assert validate(str(content)), "Unexpected response"
            report = {"check": name, "status": "PASS"}
        except Exception as error:
            failures += 1
            report = {"check": name, "status": "FAIL", "error_type": type(error).__name__}
            status_code = getattr(error, "status_code", None)
            if isinstance(status_code, int):
                report["http_status"] = status_code
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        print(json.dumps(report), flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
