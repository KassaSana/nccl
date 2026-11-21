#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# memtrace run matrix on a Linux notebook VM (written for Kaggle 2x T4, works on any Linux box).
# 1. Records the environment (same fields as kaggle_validate.sh).
# 2. Builds NCCL $NCCL_REF for the detected GPU arch only, optionally with $NCCL_PATCH applied. A build in
#    $WORK is reused when present.
# 3. Builds memtrace.cc and runs it for each (nGpus, ranksPerGpu) config and NCCL_RUNTIME_CONNECT in {1, 0}.
#    ranksPerGpu > 1 puts ranks on the same GPU, so they use the P2P transport with device buffers; with one
#    rank per T4 NCCL uses SHM (host memory) and per-step device deltas stay at 0.
# 4. Runs nccl_memacct.py on each run and writes everything to $OUT, plus a tarball.
#
# Usage: bash kaggle_memtrace.sh
# Env overrides: NCCL_REF (tag/branch), NCCL_REPO, NCCL_PATCH (git diff applied after clone), OUT, WORK,
#   SPEC (memtrace ops), CONFIGS (space-separated "nGpus x ranksPerGpu" pairs, default "2x1 2x2"),
#   RUN_TIMEOUT (seconds per run, default 300; exit 124 means it was killed), JOBS.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$(cd "$HERE/.." && pwd)"
NCCL_REF="${NCCL_REF:-v2.32.6-1}"  # the version behind the #2538 results
NCCL_REPO="${NCCL_REPO:-https://github.com/NVIDIA/nccl.git}"
NCCL_PATCH="${NCCL_PATCH:-}"
if [ -d /kaggle/working ]; then DEFAULT_ROOT=/kaggle/working; else DEFAULT_ROOT="$PWD"; fi
OUT="${OUT:-$DEFAULT_ROOT/memtrace_results}"
# Builds stay out of /kaggle/working: everything there becomes kernel output, and thousands of build files make
# "kaggle kernels output" slow and rate limited.
WORK="${WORK:-/tmp/memacct_work}"
SPEC="${SPEC:-sendrecv:1M,allreduce:1M,allreduce:64M,allreduce:1M,reducescatter:1M,a2a_sendrecv:1M,alltoall:1M}"
# 1x2 (two ranks on one GPU) is opt-in: on a T4 it hangs at allreduce:64M. It runs 64 channels on 40 SMs; unverified guess: both ranks'
# kernels cannot be resident at once. Add "1x2" with NCCL_MAX_NCHANNELS set low to try it.
CONFIGS="${CONFIGS:-2x1 2x2}"
RUN_TIMEOUT="${RUN_TIMEOUT:-300}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

mkdir -p "$OUT" "$WORK"
ENV_REPORT="$OUT/environment.txt"
SUMMARY="$OUT/summary.md"

die() {
  echo "ERROR: $*" | tee -a "$ENV_REPORT" >&2
  exit 1
}

for f in memtrace.cc nccl_memacct.py; do
  [ -f "$TOOL/$f" ] || die "missing $TOOL/$f (copy the whole nccl_memacct folder, including validation/)"
done

# ---------------------------------------------------------------- 1. environment
{
  echo "# memtrace validation environment"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nccl_ref_requested: $NCCL_REF"
  echo "nccl_repo: $NCCL_REPO"
  echo "nccl_patch: ${NCCL_PATCH:-none}"
  echo "os: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)"
  echo "kernel: $(uname -r)"
  echo "cpus: $JOBS"
} > "$ENV_REPORT"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found (on Kaggle, set Accelerator to a GPU)"
NGPU_FOUND=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$NGPU_FOUND" -ge 1 ] || die "nvidia-smi reports no GPUs"
{
  echo "gpus_found: $NGPU_FOUND"
  echo "gpu_details (index, name, compute_cap, driver, memory.total):"
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv,noheader | sed 's/^/  /'
  echo "topology:"
  nvidia-smi topo -m 2>/dev/null | sed 's/^/  /'
} >> "$ENV_REPORT"

NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] || for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do [ -x "$c" ] && NVCC="$c" && break; done
[ -n "$NVCC" ] || die "nvcc not found: a CUDA toolkit is required"
CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$NVCC")/.." && pwd)}"
{
  echo "nvcc: $NVCC"
  echo "nvcc_version: $("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  echo "gcc: $(gcc --version 2>/dev/null | head -1)"
  echo "python: $(python3 --version 2>&1)"
} >> "$ENV_REPORT"

GENCODE=""
for cc in $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' .' | sort -u); do
  GENCODE="$GENCODE -gencode=arch=compute_${cc},code=sm_${cc}"
done
echo "nvcc_gencode:$GENCODE" >> "$ENV_REPORT"

# ---------------------------------------------------------------- 2. NCCL
TAG="$NCCL_REF"
[ -n "$NCCL_PATCH" ] && TAG="$NCCL_REF+$(basename "$NCCL_PATCH" .patch)"
SRC="$WORK/nccl-${TAG//\//_}"
if [ ! -f "$SRC/build/lib/libnccl.so" ]; then
  rm -rf "$SRC"
  git clone --depth 1 --branch "$NCCL_REF" "$NCCL_REPO" "$SRC" > "$OUT/nccl_clone.log" 2>&1 \
    || die "git clone of $NCCL_REF failed (see nccl_clone.log)"
  if [ -n "$NCCL_PATCH" ]; then
    git -C "$SRC" apply --verbose "$NCCL_PATCH" > "$OUT/nccl_patch.log" 2>&1 || die "patch failed (see nccl_patch.log)"
  fi
  echo "Building NCCL $TAG with -j$JOBS (this can take 10+ minutes)..."
  make -C "$SRC" -j"$JOBS" src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$GENCODE" > "$OUT/nccl_build.log" 2>&1 \
    || die "NCCL build failed (see nccl_build.log)"
fi
NCCL_BUILD="$SRC/build"
{
  echo "nccl_commit: $(git -C "$SRC" rev-parse HEAD)"
  echo "nccl_tree_dirty: $(git -C "$SRC" status --porcelain -- src | wc -l) file(s)"
} >> "$ENV_REPORT"

# ---------------------------------------------------------------- 3. memtrace
"$NVCC" -O2 -o "$WORK/memtrace" "$TOOL/memtrace.cc" -I"$NCCL_BUILD/include" -L"$NCCL_BUILD/lib" -lnccl \
  -Xlinker -rpath="$NCCL_BUILD/lib" > "$OUT/memtrace_build.log" 2>&1 || die "memtrace build failed (see memtrace_build.log)"

# ---------------------------------------------------------------- 4. run matrix
{
  echo "# memtrace summary"
  echo
  echo "NCCL $TAG, spec \`$SPEC\`. Snapshots are taken after synchronization and are not guaranteed settled"
  echo "states. device_* columns are device-wide; tracked_* is ncclCommMemStats summed over the ranks of a device."
  echo
  echo "| run | nGpus | ranksPerGpu | NCCL_RUNTIME_CONNECT | exit |"
  echo "|---|---|---|---|---|"
} > "$SUMMARY"

unset NCCL_DEBUG_FILE  # the stage markers and NCCL log must share stdout
for cfg in $CONFIGS; do
  ng="${cfg%x*}"
  rpg="${cfg#*x}"
  if [ "$ng" -gt "$NGPU_FOUND" ]; then
    echo "Skipping $cfg: only $NGPU_FOUND GPU(s)"
    continue
  fi
  for rc in 1 0; do
    run="g${ng}_r${rpg}_rc${rc}"
    dir="$OUT/$run"
    mkdir -p "$dir"
    echo "Running $run..."
    ( cd "$dir" && env LD_LIBRARY_PATH="$NCCL_BUILD/lib:${LD_LIBRARY_PATH:-}" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC \
        NCCL_RUNTIME_CONNECT="$rc" timeout "$RUN_TIMEOUT" "$WORK/memtrace" -g "$ng" -r "$rpg" -o "$SPEC" -j memtrace.json \
        > nccl.log 2> steps.txt )
    status=$?
    echo "| $run | $ng | $rpg | $rc | $status |" >> "$SUMMARY"
    if [ "$status" -eq 0 ] && [ -f "$dir/memtrace.json" ]; then
      python3 "$TOOL/nccl_memacct.py" "$dir/nccl.log" --stats "$dir/memtrace.json" --granularity 2MiB \
        > "$dir/report.md" 2> "$dir/report_err.log" || echo "(parser failed, see report_err.log)" > "$dir/report.md"
    else
      echo "(run failed with exit $status, see steps.txt)" > "$dir/report.md"
    fi
  done
done

for run_dir in "$OUT"/g*_r*_rc*; do
  [ -d "$run_dir" ] || continue
  {
    echo
    echo "## $(basename "$run_dir")"
    echo
    echo '```'
    cat "$run_dir/steps.txt"
    echo '```'
    echo
    cat "$run_dir/report.md"
  } >> "$SUMMARY"
done

TARBALL="$(dirname "$OUT")/$(basename "$OUT").tar.gz"
tar -czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "Done. Results: $OUT (summary.md, environment.txt, per-run logs) and $TARBALL"
