#!/bin/bash
# Backwards-compat shim for the original L4-only finder. New work should call
# scripts/gcp_find_gpu_and_train.sh directly with --accelerator l4 (or another
# preset). Preserved so historical commands logged in docs/findings.tex still
# resolve.

set -uo pipefail
exec bash "$(dirname "$0")/gcp_find_gpu_and_train.sh" "$@" --accelerator l4
