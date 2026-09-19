"""Exercise production TLS/login defenses from isolated loopback test IPs.

No firewall or address changes, no paid model calls, no real-user lockout.
Credentials are read privately and are never printed or passed on a command line.
"""
import base64
import argparse
import http.client
import json
from pathlib import Path
import socket
import ssl
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-base", type=int, default=10,
                        help="Start of three unused loopback test IPs; use a new group within 15 minutes")
    parser.add_argument("--domain", default="ai.chx1008.com")
    args = parser.parse_args()
    assert 2 <= args.source_base <= 250, "Use only dedicated 127.0.0.2..252 test addresses"
    attacker, normal, flood = (f"127.0.0.{args.source_base + offset}" for offset in range(3))
    root = Path(__file__).resolve().parents[2]
    credentials = dict(line.split(": ", 1) for line in
                       (root / "deployment-secrets.txt").read_text().splitlines() if ": " in line)
    def basic(password):
        return "Basic " + base64.b64encode((credentials["Username"] + ":" + password).encode()).decode()
    valid = basic(credentials["Password"])
    invalid = basic("invalid-security-check-" + uuid.uuid4().hex)
    context = ssl.create_default_context()

    def request(ip, path, *, authorization=None, data=None, spoof=False):
        connection = http.client.HTTPSConnection(args.domain, timeout=20, context=context)
        raw = socket.create_connection(("127.0.0.1", 443), timeout=20, source_address=(ip, 0))
        connection.sock = context.wrap_socket(raw, server_hostname=args.domain)
        headers = {}
        if authorization:
            headers["Authorization"] = authorization
        if data is not None:
            headers["Content-Type"] = "application/json"
        if spoof:
            headers.update({"X-Forwarded-For": "198.51.100.80", "X-Real-IP": "198.51.100.81",
                            "Forwarded": "for=198.51.100.82", "CF-Connecting-IP": "198.51.100.83"})
        try:
            connection.request("POST" if data is not None else "GET", path, body=data, headers=headers)
            response = connection.getresponse()
            result = response.status, response.getheader("Retry-After"), response.getheader("Cache-Control")
            response.read()
            return result
        finally:
            connection.close()

    # nginx -s reload returns before old workers stop accepting new connections.
    # Wait for the new protected response before counting password failures.
    deadline = time.monotonic() + 15
    while True:
        status, _, cache_control = request(normal, "/manage")
        if status == 401 and cache_control == "no-store":
            break
        assert time.monotonic() < deadline, "New login guard has not become ready"
        time.sleep(0.5)
    for path in ("/manage", "/api/admin/profile?tenant_id=default", "/api/admin/knowledge/files?tenant_id=default"):
        assert request(normal, path, authorization=valid)[0] == 200, "Valid management login failed"
    assert request(attacker, "/manage")[0] == 401, "Anonymous challenge missing"
    failures = []
    for index in range(10):
        path = "/manage" if index % 2 == 0 else "/api/admin/profile?tenant_id=default"
        status, _, _ = request(attacker, path, authorization=invalid)
        failures.append(status)
        assert status == 401, f"Expected authentication failure before threshold, got {status}"

    blocked = []
    for path in ("/manage", "/api/admin/profile?tenant_id=default",
                 "/api/%61dmin/profile?tenant_id=default", "/api//admin/profile?tenant_id=default"):
        status, retry, _ = request(attacker, path, authorization=valid, spoof=True)
        assert status == 429 and retry and 840 <= int(retry) <= 900, f"Ban failed: {path}, status={status}, retry={retry}"
        blocked.append({"path": path, "status": status, "retry_after": int(retry)})
    assert request(normal, "/manage", authorization=valid)[0] == 200, "Ban affected another source IP"
    assert request(attacker, "/")[0] == 200, "Ban affected the public website"
    assert request(attacker, "/api/health")[0] == 200, "Ban affected health checks"
    assert request(attacker, "/api/chat/stream", data=b'{')[0] == 422, "Ban affected public chat"

    rate_codes = [request(flood, "/manage")[0] for _ in range(14)]
    assert 401 in rate_codes and 429 in rate_codes and set(rate_codes) <= {401, 429}, "Management request rate limit failed"
    assert request(flood, "/api/health")[0] == 200, "Rate limit affected public health"
    assert request("127.0.0.1", "/manage")[0] == 401, "Normal monitoring address was locked out"
    print(json.dumps({"status": "PASS", "failed_password_responses": failures,
                      "blocked_management_routes": blocked, "anonymous_rate_limit": rate_codes,
                      "other_ip_login_ok": True, "public_chat_unaffected": True, "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error_type": type(error).__name__,
                          "assertion": str(error) if isinstance(error, AssertionError) else None}))
        raise SystemExit(1)
