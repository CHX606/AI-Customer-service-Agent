"""Validate this site's real Cloudflare path without printing credentials/data.

--chat additionally makes one ordinary paid-model chat request in a fresh test
session. Nothing is changed in the knowledge base, profile, DNS, or firewall.
"""
import argparse
import base64
import ipaddress
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import uuid


BASE = "https://ai.chx1008.com"
ROOT = Path(__file__).resolve().parents[2]
LOG = Path("/opt/1panel/www/sites/ai-customer-service/log/access.log")


class NoRedirect(HTTPRedirectHandler):
    # In particular, never forward the management Authorization header through
    # a misconfigured cross-host redirect.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({}), NoRedirect())
    report = {}

    def request(url, *, authorization=None):
        headers = {"User-Agent": "ACS-deployment-check/1.0"}
        if authorization:
            headers["Authorization"] = authorization
        try:
            response = opener.open(Request(url, headers=headers), timeout=30)
        except HTTPError as error:
            response = error
        with response:
            return response.code, response.headers, response.read()

    def check(condition, label):
        assert condition, label
        report[label] = True
        print("PASS: " + label, flush=True)

    redirect_path = "/manage?edge_check=1"
    status, headers, _ = request(BASE.replace("https:", "http:") + redirect_path)
    check(status in (301, 308) and headers.get("Location") == BASE + redirect_path,
          "HTTP redirects to HTTPS preserving path and query")

    status, headers, html = request(BASE + "/")
    check(status == 200 and b'<div id="root"' in html and headers.get("CF-Ray"),
          "frontend served through Cloudflare with valid TLS")
    scripts = re.findall(rb'<script[^>]+src="([^"]+)"', html)
    check(any(asset.startswith(b"/assets/") for asset in scripts), "frontend references JavaScript assets")
    for asset in scripts:
        if not asset.startswith(b"/assets/"):
            source = urlsplit(asset.decode())
            check(source.scheme == "https" and source.netloc == "static.cloudflareinsights.com"
                  and source.path.startswith("/beacon.min.js"), "additional script is Cloudflare's analytics beacon")
            continue
        check(asset.startswith(b"/assets/"), "JavaScript asset uses same-origin path")
        status, _, body = request(BASE + asset.decode())
        check(status == 200 and b"/api" in body, "JavaScript asset loads with same-origin API")

    status, _, trace = request(BASE + "/cdn-cgi/trace")
    trace_values = dict(line.split("=", 1) for line in trace.decode().splitlines() if "=" in line)
    check(status == 200 and "ip" in trace_values, "Cloudflare reports this probe's public client IP")
    expected_client = ipaddress.ip_address(trace_values["ip"])
    log_offset = LOG.stat().st_size
    status, headers, body = request(BASE + "/api/health")
    check(status == 200 and json.loads(body).get("status") == "ok", "public backend health")
    check("no-store" in headers.get("Cache-Control", "")
          and headers.get("CF-Cache-Status", "").upper() not in ("HIT", "STALE", "UPDATING"),
          "API is not cached at the edge")
    trusted = [ipaddress.ip_network(value) for value in re.findall(
        r"^set_real_ip_from\s+([^;]+);", (ROOT / "infra/deploy/cloudflare-realip.conf").read_text(), re.M)]
    with LOG.open() as handle:
        handle.seek(log_offset)
        lines = handle.read().splitlines()
    verified = False
    for line in lines:
        if '"GET /api/health ' not in line:
            continue
        match = re.match(r"(\S+) peer=(\S+)", line)
        if match:
            client, peer = (ipaddress.ip_address(value) for value in match.groups())
            verified |= client == expected_client and client != peer and any(peer in net for net in trusted)
    check(verified, "origin restores the visitor IP and records a trusted Cloudflare peer")

    for path in ("/manage", "/api/admin/profile?tenant_id=default"):
        status, headers, _ = request(BASE + path)
        check(status == 401 and "no-store" in headers.get("Cache-Control", ""),
              "anonymous management rejected without caching: " + path)

    credentials = dict(line.split(": ", 1) for line in
                       (ROOT / "deployment-secrets.txt").read_text().splitlines() if ": " in line)
    authorization = "Basic " + base64.b64encode(
        (credentials["Username"] + ":" + credentials["Password"]).encode()).decode()
    for path in ("/manage", "/api/admin/profile?tenant_id=default", "/api/admin/knowledge/files?tenant_id=default"):
        status, headers, body = request(BASE + path, authorization=authorization)
        check(status == 200 and "no-store" in headers.get("Cache-Control", ""),
              "authenticated management works through Cloudflare: " + path)
    for path in ("/manage", "/api/admin/profile?tenant_id=default"):
        status, _, _ = request(BASE + path)
        check(status == 401, "authenticated content is not reused for anonymous visitors: " + path)

    if args.chat:
        payload = json.dumps({"message": "续费之后为什么流量没有重置？", "tenant_id": "default",
                              "session_id": "cf-acceptance-" + uuid.uuid4().hex}).encode()
        start = time.monotonic()
        first = None
        events = []
        with opener.open(Request(BASE + "/api/chat/stream", data=payload,
                                  headers={"Content-Type": "application/json",
                                           "User-Agent": "ACS-deployment-check/1.0"}), timeout=180) as response:
            check(response.status == 200 and response.headers.get("CF-Ray"), "chat reaches backend through Cloudflare")
            for line in response:
                if line.strip():
                    if first is None:
                        first = time.monotonic() - start
                    events.append(json.loads(line))
        duration = time.monotonic() - start
        check(any(event.get("type") == "final" and event.get("answer") for event in events)
              and not any(event.get("type") == "error" for event in events), "real streamed chat finishes successfully")
        report["chat_first_event_seconds"] = round(first, 3)
        report["chat_total_seconds"] = round(duration, 3)
        report["chat_event_types"] = sorted({event.get("type", "unknown") for event in events})
    print(json.dumps({"status": "PASS", **report}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error_type": type(error).__name__,
                          "assertion": str(error) if isinstance(error, AssertionError) else None}))
        raise SystemExit(1)
