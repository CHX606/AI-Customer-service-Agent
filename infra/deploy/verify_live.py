"""Real local acceptance checks; preserve a small checkpoint across restarts.

prepare changes a profile field and uploads a uniquely named test document.
verify checks persistence after the operator restarts this project's services.
cleanup restores that field and deletes only this check's uploaded source.
Credentials and API keys are never printed.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
import uuid

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "deployment-acceptance-state.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("phase", choices=("prepare", "verify", "cleanup", "reindex"))
    args = parser.parse_args()
    base = args.url.rstrip("/")
    credentials = dict(line.split(": ", 1) for line in
                       (ROOT / "deployment-secrets.txt").read_text().splitlines() if ": " in line)
    authorization = "Basic " + base64.b64encode(
        (credentials.get("Username", "admin") + ":" + credentials["Password"]).encode()
    ).decode()
    opener = build_opener(ProxyHandler({}))

    def request(path, *, method=None, data=None, admin=False, content_type=None):
        headers = {}
        if admin:
            headers["Authorization"] = authorization
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False).encode()
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        req = Request(base + path, data=data, headers=headers, method=method)
        try:
            with opener.open(req, timeout=300) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def api(path, **kwargs):
        status, body = request(path, **kwargs)
        assert status == 200, f"{path}: HTTP {status}"
        return json.loads(body)

    def save(state):
        descriptor = os.open(STATE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)

    def stream(message, session_id):
        payload = json.dumps({"message": message, "tenant_id": "default",
                              "session_id": session_id}).encode()
        started = time.monotonic()
        events = []
        timings = []
        req = Request(base + "/api/chat/stream", data=payload,
                      headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=300) as response:
            assert response.status == 200
            assert "application/x-ndjson" in response.headers.get("Content-Type", "")
            for line in response:
                if line.strip():
                    events.append(json.loads(line))
                    timings.append(round(time.monotonic() - started, 3))
        assert events and not any(event.get("type") == "error" for event in events)
        final = next(event for event in reversed(events) if event.get("type") == "final")
        assert final.get("answer")
        assert timings[0] < timings[-1], "No streaming progress before completion"
        return {"answer": final["answer"], "event_types": [event["type"] for event in events],
                "first_event_seconds": timings[0], "total_seconds": timings[-1]}

    profile_path = "/api/admin/profile?tenant_id=default"
    files_path = "/api/admin/knowledge/files?tenant_id=default"
    if args.phase == "reindex":
        for source_id in ("legacy_kelecloud_docx", "builtin_external_apps_guide"):
            source = api(f"/api/admin/knowledge/files/{source_id}/reindex?tenant_id=default", admin=True, method="POST")
            assert source["status"] == "ready" and source["chunk_count"] > 0
            print(json.dumps({"check": "admin_reindex", "status": "PASS", "source_id": source_id,
                              "chunks": source["chunk_count"]}), flush=True)
        return
    if args.phase == "prepare":
        assert not STATE.exists(), "Existing checkpoint: verify/cleanup it before another prepare"
        assert request("/")[0] == 200
        assert api("/api/health")["status"] == "ok"
        assert api("/api/health")["features"]["image_chat"] is True
        assert api("/api/public/profile?tenant_id=default")["tenant_id"] == "default"
        for path in ("/manage", profile_path, files_path):
            assert request(path)[0] == 401, f"Anonymous access accepted: {path}"
        assert request("/manage", admin=True)[0] == 200
        profile = api(profile_path, admin=True)
        state = {"marker": "ACS-DEPLOY-" + uuid.uuid4().hex[:12],
                 "original_business_hours": profile.get("business_hours")}
        state["session_id"] = "deploy-smoke-" + state["marker"]
        state["filename"] = state["marker"] + ".md"
        save(state)
        api("/api/admin/profile", method="PUT", admin=True,
            data={"tenant_id": "default", "business_hours": state["marker"]})
        content = ("# 可乐云部署验收\n\n可乐云本次部署验收专用编号是 " + state["marker"] +
                   "。这是一条临时验收资料，验收后删除。\n").encode()
        boundary = "acceptance-" + uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"tenant_id\"\r\n\r\ndefault\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{state['filename']}\"\r\n"
                "Content-Type: text/markdown\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
        uploaded = api("/api/admin/knowledge/files", admin=True, data=body,
                       content_type="multipart/form-data; boundary=" + boundary)
        state["source_id"] = uploaded["source_id"]
        save(state)
        assert uploaded["status"] == "ready" and uploaded["chunk_count"] > 0
        assert any(source["source_id"] == state["source_id"] for source in api(files_path, admin=True))
        state["grounded_chat"] = stream("续费之后为什么流量没有重置？", state["session_id"])
        answer = state["grounded_chat"]["answer"]
        assert "续费" in answer and "重置" in answer
        assert any(word in answer for word in ("有效期", "延长", "期限")), "Answer did not explain the documented distinction"
        save(state)
        print(json.dumps({"status": "PASS", "checks": ["frontend", "health", "profile", "anonymous_rejection",
              "admin_authentication", "profile_update", "upload_index", "grounded_streaming_chat"],
              "stream": {key: value for key, value in state["grounded_chat"].items() if key != "answer"}}, ensure_ascii=False))
    else:
        state = json.loads(STATE.read_text())
        if args.phase == "verify":
            assert api("/api/health")["status"] == "ok"
            assert api(profile_path, admin=True)["business_hours"] == state["marker"]
            sources = api(files_path, admin=True)
            assert any(source["source_id"] == state["source_id"] and source["status"] == "ready" for source in sources)
            print("PASS: profile and uploaded knowledge survived the restart; also inspect OpenSearch and session persistence")
        else:
            api("/api/admin/profile", method="PUT", admin=True,
                data={"tenant_id": "default", "business_hours": state["original_business_hours"]})
            sources = api(files_path, admin=True)
            for source in sources:
                if source["original_filename"] == state["filename"]:
                    api("/api/admin/knowledge/files/" + source["source_id"] + "?tenant_id=default", method="DELETE", admin=True)
            assert api(profile_path, admin=True)["business_hours"] == state["original_business_hours"]
            assert not any(source["original_filename"] == state["filename"] for source in api(files_path, admin=True))
            state["cleanup_complete"] = True
            save(state)
            print("PASS: original profile restored and this check's temporary knowledge removed")


if __name__ == "__main__":
    main()
