"""Export only this site's 1Panel certificate and reload safely after renewal.

1Panel remains responsible for DNS-01 issuance/renewal. No API is enabled and no
DNS credential is read. Certificate pairs are validated before an atomic symlink
switch; a failed nginx check/reload restores the previous pair.
"""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
import uuid

DOMAIN = "ai.chx1008.com"
DATABASE = Path("/opt/1panel/db/agent.db")
DESTINATION = Path("/opt/1panel/www/sites/ai-customer-service/ssl")
NGINX = ["/usr/bin/docker", "exec", "1Panel-openresty-gYvn", "nginx"]


def run(arguments):
    return subprocess.run(arguments, check=True, capture_output=True, timeout=30).stdout


def private_write(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)


def validate_pair(directory):
    certificate, private_key = directory / "fullchain.pem", directory / "privkey.pem"
    run(["openssl", "x509", "-in", str(certificate), "-noout", "-checkhost", DOMAIN])
    run(["openssl", "x509", "-in", str(certificate), "-noout", "-checkend", "86400"])
    run(["openssl", "verify", "-verify_hostname", DOMAIN, "-CAfile", "/etc/ssl/certs/ca-certificates.crt",
         "-untrusted", str(certificate), str(certificate)])
    public_from_cert = run(["openssl", "x509", "-in", str(certificate), "-pubkey", "-noout"])
    public_from_key = run(["openssl", "pkey", "-in", str(private_key), "-pubout"])
    if public_from_cert != public_from_key:
        raise RuntimeError("Certificate and private key do not match")


def switch_link(link, target):
    temporary = link.with_name(".current-" + uuid.uuid4().hex)
    temporary.symlink_to(target)
    temporary.replace(link)


def synchronize(*, check_only=False, bootstrap=False):
    with sqlite3.connect(DATABASE.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT id,pem,private_key,auto_renew FROM website_ssls "
            "WHERE primary_domain=? AND status='ready' ORDER BY id DESC LIMIT 1", (DOMAIN,),
        ).fetchone()
    if row is None:
        raise RuntimeError("No ready certificate for the configured domain")
    certificate_id, certificate, private_key, renewal = row
    if not renewal or not certificate.startswith("-----BEGIN CERTIFICATE-----") or not private_key.startswith("-----BEGIN "):
        raise RuntimeError("Certificate material or automatic renewal is unavailable")

    with TemporaryDirectory(prefix="acs-certificate-") as temporary:
        temporary = Path(temporary)
        private_write(temporary / "fullchain.pem", certificate.encode())
        private_write(temporary / "privkey.pem", private_key.encode())
        validate_pair(temporary)
        if check_only:
            print(f"PASS: certificate {certificate_id} for {DOMAIN}; chain, hostname, expiry and key verified")
            return

        DESTINATION.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(DESTINATION, 0o700)
        versions = DESTINATION / "versions"
        versions.mkdir(mode=0o700, exist_ok=True)
        version = versions / sha256(certificate.encode()).hexdigest()[:24]
        version.mkdir(mode=0o700, exist_ok=True)
        for name in ("fullchain.pem", "privkey.pem"):
            value = (temporary / name).read_bytes()
            target = version / name
            if target.exists():
                if target.read_bytes() != value:
                    raise RuntimeError("Existing certificate version has unexpected content")
            else:
                private_write(target, value)

        current = DESTINATION / "current"
        if current.exists() and not current.is_symlink():
            raise RuntimeError("Refusing to replace a non-symlink certificate location")
        previous = os.readlink(current) if current.is_symlink() else None
        next_target = str(version.relative_to(DESTINATION))
        if previous == next_target:
            print(f"OK: certificate {certificate_id} already synchronized")
            return
        if bootstrap and previous is not None:
            raise RuntimeError("Bootstrap mode is only allowed for the first installation")
        switch_link(current, next_target)
        if not bootstrap:
            try:
                run([*NGINX, "-t"])
                run([*NGINX, "-s", "reload"])
            except Exception:
                if previous is not None:
                    switch_link(current, previous)
                else:
                    current.unlink()
                raise
        print(f"PASS: certificate {certificate_id} synchronized for {DOMAIN}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    try:
        synchronize(check_only=args.check_only, bootstrap=args.bootstrap)
    except Exception as error:
        # Never emit database content, certificate keys or subprocess output.
        print(f"Certificate synchronization failed: {type(error).__name__}")
        raise SystemExit(1)
