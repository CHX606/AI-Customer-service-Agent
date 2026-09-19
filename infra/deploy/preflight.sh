#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

[[ $(uname -s) == Linux ]] || { echo 'Requires a Linux Docker host.'; exit 1; }
[[ $(uname -m) == x86_64 ]] || { echo 'This package targets Linux amd64. ARM needs separate wheel/model validation.'; exit 1; }
grep -qw avx2 /proc/cpuinfo || { echo 'The supplied INT8 configuration requires AVX2.'; exit 1; }
command -v python3 >/dev/null
docker info >/dev/null
docker compose version
memory_kb=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
disk_kb=$(df -Pk . | awk 'NR==2 {print $4}')
echo "Memory: $((memory_kb / 1024)) MiB; workspace disk available: $((disk_kb / 1024)) MiB"
if (( memory_kb < 15000000 )); then
    echo 'Capacity review required: 16 GiB RAM is recommended for OpenSearch, ONNX export and OCR together.'
fi
if (( disk_kb < 40000000 )); then
    echo 'Capacity review required: reserve about 40 GB for image builds and model files.'
fi
maps=$(cat /proc/sys/vm/max_map_count)
if (( maps < 262144 )); then
    echo 'Set vm.max_map_count to at least 262144, persist it via sysctl.d, then rerun.'
    exit 1
fi
echo 'Check Docker data-root disk space, existing websites and occupied ports before deployment.'
echo 'Preflight completed; model inference capacity still requires live measurement.'
