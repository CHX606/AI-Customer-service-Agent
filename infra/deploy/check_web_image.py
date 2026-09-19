"""Check the real web image on loopback; this does not validate the backend."""
import base64
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
KEYS = {"SITE_ADDRESS", "MANAGEMENT_USER", "MANAGEMENT_PASSWORD_HASH", "ADMIN_API_KEY"}


def main():
    config = {}
    for line in (ROOT / ".env.production").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key in KEYS:
            value = value.strip()
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            config[key] = value
    if not all(config.get(key) for key in KEYS):
        raise SystemExit("Missing web configuration")
    # Always test HTTP on loopback, even if production is later given a domain.
    config["SITE_ADDRESS"] = ":80"
    credentials = dict(
        line.split(": ", 1)
        for line in (ROOT / "deployment-secrets.txt").read_text().splitlines()
        if ": " in line
    )
    authorization = "Basic " + base64.b64encode(
        (config["MANAGEMENT_USER"] + ":" + credentials["Password"]).encode()
    ).decode()
    environment = os.environ.copy()
    environment.update(config)
    command = [
        "docker", "run", "--rm", "--detach", "--name", "ai-customer-service-web-check",
        "--memory=128m", "--memory-swap=256m", "--cpus=0.5",
        "--publish", "127.0.0.1:18080:80",
        "--env", "BACKEND_UPSTREAM=127.0.0.1:9",
    ]
    for key in sorted(KEYS):
        command.extend(["--env", key])
    command.append("ai-customer-service-web:local")
    container = subprocess.run(command, env=environment, capture_output=True, text=True, check=True).stdout.strip()
    opener = build_opener(ProxyHandler({}))

    def request(path, headers=None):
        try:
            with opener.open(Request("http://127.0.0.1:18080" + path, headers=headers or {}), timeout=30) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    try:
        for attempt in range(20):
            try:
                status, html = request("/")
                break
            except (URLError, ConnectionError, TimeoutError):
                if attempt == 19:
                    raise
                time.sleep(0.5)
        assert status == 200 and b'<div id="root"' in html, "Frontend HTML failed"
        print("PASS: real frontend HTML over loopback")
        scripts = re.findall(rb'<script[^>]+src="([^"]+)"', html)
        assert scripts, "Missing built JavaScript"
        for asset in scripts:
            assert asset.startswith(b"/assets/")
            status, body = request(asset.decode())
            assert status == 200 and b"/api" in body, "Same-origin API build failed"
            for secret in (config["ADMIN_API_KEY"], credentials["Password"]):
                assert secret.encode() not in body, "Secret found in frontend asset"
        print("PASS: built JavaScript uses /api; server secrets absent")
        for path in ("/manage", "/api/admin/profile?tenant_id=default"):
            assert request(path)[0] == 401, "Anonymous management was not denied"
        assert request("/api/admin/profile?tenant_id=default", {"X-Admin-Key": config["ADMIN_API_KEY"]})[0] == 401
        print("PASS: anonymous and forged-header management requests denied")
        status, body = request("/manage", {"Authorization": authorization})
        assert status == 200 and b'<div id="root"' in body, "Management password failed"
        print("PASS: generated management credentials accepted by Caddy")
        print("Frontend/proxy checks passed. Backend, models, search, TLS and browser checks remain pending.")
    finally:
        subprocess.run(["docker", "stop", "--time", "5", container], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
