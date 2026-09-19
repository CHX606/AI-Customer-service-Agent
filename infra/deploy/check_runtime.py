"""Read-only, secret-free runtime checks for this deployed customer-service stack."""
import json
import subprocess
import sys

NAMES = tuple("ai-customer-service-" + service + "-1" for service in ("web", "backend", "opensearch"))
WEB_PORTS = {
    "80/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18080"}],
    "443/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18443"}],
}


def validate(containers, search_internal):
    errors = []
    by_name = {item["name"].lstrip("/"): item for item in containers}
    for name in NAMES:
        item = by_name.get(name)
        if item is None:
            errors.append(f"customer-service container missing: {name}")
            continue
        if not item["running"]:
            errors.append(f"customer-service container not running: {name}")
        if item["privileged"] or item["pid_mode"] == "host":
            errors.append(f"customer-service host privilege exposed: {name}")
        if "/var/run/docker.sock" in item["mounts"] or "/run/docker.sock" in item["mounts"]:
            errors.append(f"Docker socket mounted into customer-service: {name}")
        if item["oom"]:
            errors.append(f"customer-service container OOM: {name}")
        if name.endswith("-web-1"):
            if item["ports"] != WEB_PORTS:
                errors.append("customer-service web ports are not the approved loopback-only bindings")
            networks = {"ai-customer-service_web"}
        else:
            if item["ports"]:
                errors.append(f"customer-service internal port published: {name}")
            if item["health"] != "healthy":
                errors.append(f"customer-service container unhealthy: {name}")
            if item["user"].split(":", 1)[0] in ("", "0", "root"):
                errors.append(f"customer-service application runs as root: {name}")
            networks = {"ai-customer-service_search"}
            if name.endswith("-backend-1"):
                networks.add("ai-customer-service_web")
                if "no-new-privileges:true" not in (item["security"] or []):
                    errors.append("customer-service backend no-new-privileges missing")
        if set(item["networks"]) != networks:
            errors.append(f"customer-service unexpected network attachment: {name}")
    if search_internal is not True:
        errors.append("customer-service search network is no longer internal")
    return errors


def main():
    # Explicit projection: never retrieve or print container environment secrets.
    template = ('{"name":{{json .Name}},"user":{{json .Config.User}},'
                '"running":{{json .State.Running}},"oom":{{json .State.OOMKilled}},'
                '"health":{{if .State.Health}}{{json .State.Health.Status}}{{else}}null{{end}},'
                '"privileged":{{json .HostConfig.Privileged}},"pid_mode":{{json .HostConfig.PidMode}},'
                '"ports":{{json .HostConfig.PortBindings}},"security":{{json .HostConfig.SecurityOpt}},'
                '"networks":{{json .NetworkSettings.Networks}},"mounts":[{{range $i,$m := .Mounts}}'
                '{{if $i}},{{end}}{{json $m.Destination}}{{end}}]}')
    try:
        result = subprocess.run(["docker", "inspect", "--format", template, *NAMES],
                                capture_output=True, text=True, timeout=20, check=True)
        containers = [json.loads(line) for line in result.stdout.splitlines() if line]
        network = subprocess.run(["docker", "network", "inspect", "--format", "{{json .Internal}}",
                                  "ai-customer-service_search"],
                                 capture_output=True, text=True, timeout=20, check=True)
        errors = validate(containers, json.loads(network.stdout))
    except Exception as error:
        print("customer-service runtime inspection failed: " + type(error).__name__)
        return 1
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
