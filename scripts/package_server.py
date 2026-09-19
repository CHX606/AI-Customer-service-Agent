"""Build a source-only server ZIP from an explicit allowlist, with SHA-256 manifest."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {"AGENTS.md", "README.md", "DEPLOY.md", "CODEX_DEPLOY.md", "DEPLOYMENT_CHECKS.md",
             ".gitignore", ".gitattributes", ".dockerignore", ".env.example", "pyproject.toml",
             "requirements.txt", "requirements-dev.txt", "compose.deploy.yaml",
             "compose.low-memory.yaml", "API_ONLY_DEPLOYMENT.md", "SERVER_DEPLOYMENT.md",
             "SYNC_GUIDE.md"}
TOP_DIRS = {"back", "front", "resources", "scripts", "infra", "docs", "tests", "evaluation"}
EXCLUDED = {"node_modules", "dist", "build", "__pycache__", ".pytest_cache", ".git", "data", "output",
            ".venv", "venv", ".idea", ".vscode", "tmp"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tsbuildinfo", ".log", ".db", ".sqlite", ".sqlite3",
                     ".zip", ".onnx", ".safetensors", ".pt", ".pth", ".ckpt", ".bin",
                     ".pem", ".key", ".p12", ".pfx", ".bak", ".before"}


def allowed_path(name):
    relative = PurePosixPath(name)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        return False
    if name not in TOP_FILES and relative.parts[0] not in TOP_DIRS:
        return False
    if any(part in EXCLUDED or part.startswith((".pytest_tmp", "domain-change-", "security-change-"))
           for part in relative.parts):
        return False
    if any(part.startswith(".env") and part != ".env.example" for part in relative.parts):
        return False
    if relative.name in {"deployment-secrets.txt", "deployment-acceptance-state.json"}:
        return False
    if name.startswith("evaluation/results/") and relative.name != ".gitkeep":
        return False
    if relative.suffix in EXCLUDED_SUFFIXES or relative.name.endswith((".db-wal", ".db-shm")):
        return False
    return name not in {"front/vite.config.js", "front/vite.config.d.ts"}


def source_paths(allow_unversioned=False):
    probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT,
                           capture_output=True, text=True)
    if probe.returncode == 0 and probe.stdout.strip() == "true":
        raw = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        return set(raw.decode("utf-8").split("\0")) - {""}, revision
    if not allow_unversioned:
        raise SystemExit("Not a Git worktree; use --allow-unversioned for an extracted source tree.")
    paths = {name for name in TOP_FILES if (ROOT / name).exists()}
    for directory in TOP_DIRS:
        top = ROOT / directory
        if top.is_symlink():
            raise SystemExit(f"Refusing linked source directory: {directory}")
        if top.is_dir():
            paths.update(path.relative_to(ROOT).as_posix() for path in top.rglob("*") if path.is_file())
    return paths, None


def collect_files(paths):
    files = {}
    for name in sorted(paths):
        if not allowed_path(name):
            continue
        relative = PurePosixPath(name)
        source = ROOT.joinpath(*relative.parts)
        if not source.is_file():
            continue
        for component in [source, *source.parents]:
            if component == ROOT:
                break
            attrs = getattr(component.lstat(), "st_file_attributes", 0)
            if component.is_symlink() or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise SystemExit(f"Refusing linked/reparse path: {name}")
        content = source.read_bytes()
        if relative.suffix == ".sh":
            content = content.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")
        files[name] = content
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/server-deploy")
    parser.add_argument("--allow-unversioned", action="store_true",
                        help="Package an extracted tree using the same explicit source allowlist.")
    parser.add_argument("--archive-name", default="ai-customer-service-server.zip")
    args = parser.parse_args()
    if (Path(args.archive_name).name != args.archive_name or "/" in args.archive_name
            or "\\" in args.archive_name or not args.archive_name.endswith(".zip")):
        parser.error("--archive-name must be a .zip filename, not a path")
    paths, revision = source_paths(args.allow_unversioned)
    files = collect_files(paths)
    for required in ("DEPLOY.md", "CODEX_DEPLOY.md", "compose.deploy.yaml", "infra/deploy/Caddyfile",
                     "infra/deploy/.env.example", "back/api.py", "front/package-lock.json"):
        if required not in files:
            raise SystemExit(f"Missing deployment file: {required}")
    baseline = {}
    if revision is None and (ROOT / "RELEASE.json").is_file():
        baseline = json.loads((ROOT / "RELEASE.json").read_text(encoding="utf-8"))
    files["RELEASE.json"] = (json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": revision,
        "baseline_source_commit": baseline.get("source_commit") or baseline.get("baseline_source_commit"),
        "source": ("current working tree including deployment additions; not a Git-only archive" if revision
                   else "current unversioned server source tree; baseline commit is provenance only"),
        "deployment": "fresh Linux amd64 CPU / AVX2",
        "runtime_data_included": False, "secrets_included": False, "model_weights_included": False,
        "linux_container_acceptance": ("Existing-server results: API_ONLY_DEPLOYMENT.md; packaging is not a fresh-deploy test."
                                       if "API_ONLY_DEPLOYMENT.md" in files else
                                       "pending on destination server; see DEPLOYMENT_CHECKS.md"),
    }, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n" for name, content in sorted(files.items())
    ).encode("utf-8")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / args.archive_name
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo("ai-customer-service/" + name)
            info.create_system = 3
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            bundle.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad:
            raise SystemExit(f"CRC check failed: {bad}")
        for name, content in files.items():
            if bundle.read("ai-customer-service/" + name) != content:
                raise SystemExit(f"Archive content mismatch: {name}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "files": len(files), "bytes": archive.stat().st_size,
                      "sha256": digest, "crc_and_contents": "passed"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
