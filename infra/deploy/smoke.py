"""Check a deployed same-origin site; --chat explicitly makes a model request."""
import argparse
import base64
import getpass
import json
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(base, path, *, data=None, authorization=None):
    headers = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data).encode()
    if authorization:
        headers["Authorization"] = authorization
    try:
        with urlopen(Request(base + path, data=data, headers=headers), timeout=180) as res:
            return res.status, res.read()
    except HTTPError as error:
        return error.code, error.read()


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL: " + message)
    print("PASS: " + message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Site origin, e.g. https://chat.example.com")
    parser.add_argument("--admin", action="store_true", help="Prompt for management credentials")
    parser.add_argument("--chat", action="store_true", help="Call the real streaming chat API")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    status, body = request(base, "/")
    require(status == 200 and b'<div id="root"' in body, "frontend HTML")
    status, body = request(base, "/api/health")
    require(status == 200 and json.loads(body).get("status") == "ok", "backend startup health")
    status, body = request(base, "/api/public/profile?tenant_id=default")
    require(status == 200 and json.loads(body).get("tenant_id") == "default", "public tenant profile")
    for path in ("/manage", "/api/admin/profile?tenant_id=default"):
        status, _ = request(base, path)
        require(status == 401, "unauthenticated management denied: " + path)
    if args.admin:
        user = input("Management user [admin]: ").strip() or "admin"
        password = getpass.getpass("Management password: ")
        auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
        status, body = request(base, "/api/admin/profile?tenant_id=default", authorization=auth)
        require(status == 200 and json.loads(body).get("tenant_id") == "default", "authenticated admin through proxy")
    if args.chat:
        status, body = request(base, "/api/chat/stream", data={
            "message": "续费之后为什么流量没有重置？", "tenant_id": "default",
            "session_id": "deploy-smoke-" + uuid.uuid4().hex,
        })
        events = [json.loads(line) for line in body.splitlines() if line.strip()] if status == 200 else []
        require(status == 200 and any(e.get("type") == "final" and e.get("answer") for e in events)
                and not any(e.get("type") == "error" for e in events), "real streaming chat final answer")
    print("Passed. Also verify live search results, image upload and persistence per DEPLOY.md.")


if __name__ == "__main__":
    main()
