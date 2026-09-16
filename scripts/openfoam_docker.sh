#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${OPENFOAM_IMAGE:-openeuler/openfoam:2506-oe2403sp2}"
HOST_PWD="$(pwd)"
if [[ "${HOST_PWD}" == "${ROOT}"* ]]; then
  CONTAINER_PWD="/workspace${HOST_PWD#"${ROOT}"}"
else
  CONTAINER_PWD="/workspace"
fi

exec docker run --rm \
  --entrypoint /bin/bash \
  -v "${ROOT}:/workspace" \
  -w "${CONTAINER_PWD}" \
  "${IMAGE}" \
  -lc 'cmd=("$@"); set --; source /opt/OpenFOAM-v2506/etc/bashrc && exec "${cmd[@]}"' \
  bash "$@"
