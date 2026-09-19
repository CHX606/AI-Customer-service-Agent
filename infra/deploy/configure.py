"""Create private server configuration without printing generated secrets."""
import argparse
from pathlib import Path
import os
import secrets
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--caddy", help="Optional existing Caddy binary; otherwise use Docker")
    args = parser.parse_args()
    root = args.directory.resolve()
    config_path = root / ".env.production"
    credentials_path = root / "deployment-secrets.txt"
    if config_path.exists() or credentials_path.exists():
        raise SystemExit("Configuration already exists; edit it in place. Nothing overwritten.")

    password = secrets.token_urlsafe(24)
    command = ([args.caddy] if args.caddy else ["docker", "run", "--rm", "-i", "caddy:2.11.4-alpine", "caddy"])
    # Read from stdin, so the plaintext password is absent from process arguments.
    result = subprocess.run(command + ["hash-password", "--algorithm", "bcrypt"],
                            input=(password + "\n").encode("utf-8"), capture_output=True, check=True)
    hashed = result.stdout.decode("utf-8").strip()
    if not hashed.startswith("$2") or len(hashed) != 60:
        raise SystemExit("Caddy did not return a bcrypt hash; no files written.")
    values = {
        "MANAGEMENT_PASSWORD_HASH": hashed,
        "ADMIN_API_KEY": secrets.token_hex(32),
        "OPENSEARCH_PASSWORD": "Aa1!" + secrets.token_hex(24),
    }
    template = Path(__file__).with_name(".env.example").read_text(encoding="utf-8")
    lines = []
    for line in template.splitlines():
        key = line.split("=", 1)[0]
        # Single quotes preserve the dollar signs in bcrypt hashes in Compose.
        lines.append(f"{key}='{values[key]}'" if key in values else line)
    os.umask(0o077)
    with config_path.open("x", encoding="utf-8", newline="\n") as out:
        out.write("\n".join(lines) + "\n")
    with credentials_path.open("x", encoding="utf-8", newline="\n") as out:
        out.write(f"Management URL: /manage\nUsername: admin\nPassword: {password}\n")
    print(f"Created {config_path.name} and {credentials_path.name} (private files).")
    print("Fill API_KEY, BASE_URL and MODEL in .env.production before building.")


if __name__ == "__main__":
    main()
