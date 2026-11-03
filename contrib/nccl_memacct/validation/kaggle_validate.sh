#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GPU validation run for nccl_memacct on a Linux notebook VM (written for Kaggle, works on any Linux box).
# 1. Checks and records the environment: GPUs, driver, CUDA toolkit, compiler, OS.
# 2. Builds NCCL (default tag v2.32.3-1, the version used in the #2527 survey) for the detected GPU arch only.
# 3. Builds harness.cc and runs a small configuration matrix with NCCL_DEBUG_SUBSYS=ALLOC.
# 4. Runs nccl_memacct.py on each run and writes everything to $OUT, plus a tarball.
#
# Usage: bash kaggle_validate.sh
# Env overrides: NCCL_REF (tag/branch), OUT (results dir), WORK (build dir), BYTES, MAX_GPUS, JOBS.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$(cd "$HERE/.." && pwd)"
NCCL_REF="${NCCL_REF:-v2.32.3-1}"
if [ -d /kaggle/working ]; then DEFAULT_ROOT=/kaggle/working; else DEFAULT_ROOT="$PWD"; fi
OUT="${OUT:-$DEFAULT_ROOT/memacct_results}"
WORK="${WORK:-$DEFAULT_ROOT/memacct_work}"
BYTES="${BYTES:-1048576}"
MAX_GPUS="${MAX_GPUS:-2}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

mkdir -p "$OUT" "$WORK"
ENV_REPORT="$OUT/environment.txt"
SUMMARY="$OUT/summary.md"

die() {
  echo "ERROR: $*" | tee -a "$ENV_REPORT" >&2
  exit 1
}

for f in harness.cc nccl_memacct.py; do
  [ -f "$TOOL/$f" ] || die "missing $TOOL/$f (copy the whole nccl_memacct folder, including validation/)"
done

# ---------------------------------------------------------------- 1. environment
{
  echo "# nccl_memacct validation environment"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nccl_ref_requested: $NCCL_REF"
  echo "os: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)"
  echo "kernel: $(uname -r)"
  echo "cpus: $JOBS"
} > "$ENV_REPORT"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: no NVIDIA GPU/driver (on Kaggle, set Accelerator to a GPU)"
NGPU_FOUND=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$NGPU_FOUND" -ge 1 ] || die "nvidia-smi reports no GPUs"
{
  echo "gpus_found: $NGPU_FOUND"
  echo "gpu_details (index, name, compute_cap, driver, memory.total):"
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv,noheader | sed 's/^/  /'
  echo "nvidia_smi_cuda_version: $(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
  echo "topology:"
  nvidia-smi topo -m 2>/dev/null | sed 's/^/  /'
} >> "$ENV_REPORT"

NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] || for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do [ -x "$c" ] && NVCC="$c" && break; done
[ -n "$NVCC" ] || die "nvcc not found: a CUDA toolkit is required to build NCCL and the harness"
CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$NVCC")/.." && pwd)}"
{
  echo "nvcc: $NVCC"
  echo "cuda_home: $CUDA_HOME"
  echo "nvcc_version: $("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  echo "gcc: $(gcc --version 2>/dev/null | head -1)"
} >> "$ENV_REPORT"
python3 -c "import sys; assert sys.version_info >= (3, 6)" 2>/dev/null || die "python3 (>= 3.6) is required for nccl_memacct.py"
echo "python: $(python3 --version 2>&1)" >> "$ENV_REPORT"

# Build only for the GPUs present (NCCL's default list is slow to build and, for CUDA >= 12.8, has no sm_75).
GENCODE=""
for cc in $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' .' | sort -u); do
  GENCODE="$GENCODE -gencode=arch=compute_${cc},code=sm_${cc}"
done
echo "nvcc_gencode:$GENCODE" >> "$ENV_REPORT"

# ---------------------------------------------------------------- 2. NCCL
SRC="$WORK/nccl-$NCCL_REF"
if [ ! -f "$SRC/build/lib/libnccl.so" ]; then
  rm -rf "$SRC"
  git clone --depth 1 --branch "$NCCL_REF" https://github.com/NVIDIA/nccl.git "$SRC" > "$OUT/nccl_clone.log" 2>&1 \
    || die "git clone of $NCCL_REF failed (see nccl_clone.log)"
  echo "Building NCCL $NCCL_REF with -j$JOBS (this can take 10+ minutes)..."
  make -C "$SRC" -j"$JOBS" src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$GENCODE" > "$OUT/nccl_build.log" 2>&1 \
    || die "NCCL build failed (see nccl_build.log)"
fi
NCCL_BUILD="$SRC/build"
{
  echo "nccl_commit: $(git -C "$SRC" rev-parse HEAD)"
  echo "nccl_header_version: $(grep -E '#define NCCL_(MAJOR|MINOR|PATCH|SUFFIX)' "$NCCL_BUILD/include/nccl.h" | awk '{print $3}' | tr '\n' ' ')"
} >> "$ENV_REPORT"

# ---------------------------------------------------------------- 3. harness
"$NVCC" -O2 -o "$WORK/harness" "$TOOL/harness.cc" -I"$NCCL_BUILD/include" -L"$NCCL_BUILD/lib" -lnccl \
  -Xlinker -rpath="$NCCL_BUILD/lib" > "$OUT/harness_build.log" 2>&1 || die "harness build failed (see harness_build.log)"

# ---------------------------------------------------------------- 4. run matrix
NGPU_MAX=$(( NGPU_FOUND < MAX_GPUS ? NGPU_FOUND : MAX_GPUS ))
{
  echo "# nccl_memacct validation summary"
  echo
  echo "Environment: see environment.txt. NCCL $NCCL_REF; snapshots are taken after synchronization and"
  echo "are not guaranteed settled states. Tables report observed allocation requests, not live memory."
  echo
  echo "| run | nGpus | NCCL_RUNTIME_CONNECT | exit | NCCL version line |"
  echo "|---|---|---|---|---|"
} > "$SUMMARY"

unset NCCL_DEBUG_FILE  # the stage markers and NCCL log must share stdout
for ng in $(seq 1 "$NGPU_MAX"); do
  for rc in 1 0; do
    run="ng${ng}_rc${rc}"
    dir="$OUT/$run"
    mkdir -p "$dir"
    echo "Running $run..."
    ( cd "$dir" && env LD_LIBRARY_PATH="$NCCL_BUILD/lib:${LD_LIBRARY_PATH:-}" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC \
        NCCL_RUNTIME_CONNECT="$rc" timeout 900 "$WORK/harness" "$ng" "$BYTES" stats.json > nccl.log 2> stderr.log )
    status=$?
    version=$(grep -m1 -o 'NCCL version [^ ]*' "$dir/nccl.log" || echo "not found")
    echo "| $run | $ng | $rc | $status | $version |" >> "$SUMMARY"
    if [ "$status" -eq 0 ] && [ -f "$dir/stats.json" ]; then
      python3 "$TOOL/nccl_memacct.py" "$dir/nccl.log" --stats "$dir/stats.json" --granularity 2MiB \
        > "$dir/report.md" 2> "$dir/report_err.log" || echo "(parser failed, see report_err.log)" > "$dir/report.md"
    else
      echo "(run failed with exit $status, see stderr.log)" > "$dir/report.md"
    fi
  done
done

for run_dir in "$OUT"/ng*_rc*; do
  { echo; echo "## $(basename "$run_dir")"; echo; cat "$run_dir/report.md"; } >> "$SUMMARY"
done

TARBALL="$(dirname "$OUT")/$(basename "$OUT").tar.gz"
tar -czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "Done. Results: $OUT (summary.md, environment.txt, per-run logs) and $TARBALL"
