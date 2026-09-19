"""Fail on startup degradation as well as an unreachable HTTP process."""
import json
from urllib.request import urlopen

with urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
    status = json.load(response)
if status.get("status") != "ok":
    raise SystemExit("Backend startup checks are degraded")
