# nccl_memacct (proof of concept)

Reports NCCL's GPU allocation requests by allocation site and compares them with
`ncclCommMemStats()` and `cudaMemGetInfo()` at each stage of a run. Context: NVIDIA/nccl#2527,
"memory accounting tool".

This reports **observed allocation requests**. It is not a complete, authoritative breakdown of live
GPU memory by NCCL component. It uses only public APIs and NCCL's own INFO logs, and changes nothing in `src/`.

## Usage

```sh
# On a Linux machine with NVIDIA GPUs, against an NCCL build:
nvcc -o harness harness.cc -I$NCCL_HOME/include -L$NCCL_HOME/lib -lnccl
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC NCCL_RUNTIME_CONNECT=1 ./harness 2 $((1<<20)) stats.json > nccl.log

python3 nccl_memacct.py nccl.log --stats stats.json --granularity 2MiB
python3 nccl_memacct.py nccl.log --by file --format csv
```

`harness` prints a `NCCL_MEMACCT begin=<stage>` marker before each stage (init, allreduce,
reducescatter, sendrecv_alltoall), so log lines can be assigned to stages. If `NCCL_DEBUG_FILE` is set, the
markers and the NCCL log end up in different files, so leave it unset.

## memtrace: memory after each operation

`memtrace` runs a list of NCCL operations back to back on one communicator. After init and after
each step it records `cudaMemGetInfo` for each device and `ncclCommMemStats` for each communicator. Each
step's delta shows whether that operation allocated new memory or reused memory from earlier steps.
For example, send/recv followed by AllReduce adds connections, but a second AllReduce at a larger size
usually does not.

```sh
nvcc -O2 -o memtrace memtrace.cc -I$NCCL_HOME/include -L$NCCL_HOME/lib -lnccl
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC ./memtrace -g 1 -r 2 \
    -o sendrecv:1M,allreduce:1M,allreduce:64M,allreduce:1M -j memtrace.json > nccl.log
python3 nccl_memacct.py nccl.log --stats memtrace.json --granularity 2MiB
```

- **Steps.** Write each step as `op:size[:iters=N][:root=N]`. Separate steps with commas, or put one per
  line in a file passed with `-f` (`#` starts a comment).
  - Ops: `allreduce`, `reducescatter`, `allgather`, `broadcast`, `reduce`, `sendrecv`, `alltoall` and
    `a2a_sendrecv`.
  - `size` means the same as nccl-tests `-b`. It is the per-rank buffer size, split per op by the same
    `getCollByteCount` rules nccl-tests uses.
  - `sendrecv` is the nccl-tests ring: send to rank+1, receive from rank-1.
  - `alltoall` is `ncclAlltoAll` on NCCL ≥ 2.28, as in nccl-tests. `a2a_sendrecv` is the grouped
    `ncclSend`/`ncclRecv` alltoall measured in #2527.
  - `comm=` and `graph=` are reserved for multiple communicators and CUDA graphs. They are rejected for now.
- **Ranks.** `-g` sets the number of GPUs and `-r` the ranks per GPU, all in one process.
  - With `-r > 1`, memtrace sets `NCCL_MULTI_RANK_GPU_ENABLE=1`. Ranks on the same GPU then talk through the
    P2P transport with device buffers. That makes connection memory measurable on GPU pairs that otherwise
    use SHM, such as 2× T4.
  - Same-GPU P2P is not NVLink P2P.
- **What is not counted.** User buffers are sized for the largest step and allocated before the baseline,
  so no step includes them. A final `destroy` snapshot shows whether device memory returns to the
  baseline.
- **Communicator config.** All steps share one default-config communicator. nccl-tests `sendrecv_perf` sets
  `maxP2pPeers = 2` (NCCL ≥ 2.30), so its numbers can differ from a `sendrecv` step here.
- **Output.**
  - A per-step table goes to stderr. It shows device used Δ per step and cumulative, and the
    `ncclCommMemStats` total per device.
  - `memtrace.json` holds the same snapshot schema as `harness.cc`, with one snapshot per rank, plus a
    `steps` list.
  - `nccl_memacct.py` collapses ranks on the same device. It sums their stats, counting once any
    communicator marked `stats_shared_with` (a shared memory manager, see #2582).
  - The ALLOC log only carries the device, so with `-r > 1` log lines can be attributed to a device but
    not to a rank.

`validation/kaggle_memtrace.sh` builds NCCL and memtrace and runs a matrix:
- configurations `2x1` and `2x2` (GPUs × ranks per GPU)
- `NCCL_RUNTIME_CONNECT` set to 1 and to 0

`NCCL_PATCH=file.patch` applies a local NCCL change before the build.

`1x2` (two ranks on one GPU) is opt-in. On a T4 it hangs at `allreduce:64M`. That communicator uses 64
channels (its NET pool is the 64-channel size), and a T4 has 40 SMs. One guess, not yet verified, is that both
ranks' kernels can't be resident at once. Try it with a low `NCCL_MAX_NCHANNELS`.

Not covered on 2× T4 and still to be checked on other hardware:
- NVLink P2P
- NVLS
- NET with GPUDirect RDMA
- PXN
- symmetric/window kernels
- MIG

## What the numbers mean

| Column | Meaning | Caveat |
|---|---|---|
| requested | Sum of `Cuda Alloc/Calloc/CallocAsync`, `Persistent cudaMalloc[Async] work buf` and `Allocated shareable buffer` sizes. A shareable-buffer line that repeats the pointer of the line just before it (cudaMalloc path) is not counted twice | Neither a lower nor an upper bound on live memory. Some paths log nothing at INFO, and frees are never logged |
| allocation site | `file:line (func)` from the log line | A call site, not a logical component such as P2P or NVLS |
| rounded / rounding estimate | Each request rounded up to `--granularity` | An estimate. It applies only when cuMem is in use, and the real rounding happens inside NCCL |
| possible reuse | The pointer was seen before for the same host, pid and device | Possible pointer reuse, with an inferred release in between. Not a confirmed free |
| tracked_total | `ncclStatGpuMemTotal` (persist + scratch + offload) | `n/a` when the memory manager is disabled (`ncclInvalidUsage`). It excludes imported peer memory |
| tracked_minus_observed | tracked_total − cumulative requested | Expected to be nonzero. Signed, never clamped |
| device_minus_tracked | Change in device-wide used memory since `before_init`, minus tracked_total | An unexplained device-wide difference, not untracked NCCL memory |

Snapshots are taken after `cudaStreamSynchronize` and `cudaDeviceSynchronize`. Some of NCCL's asynchronous
cleanup may still be in progress, so a snapshot is not guaranteed to be a settled state.

## Known coverage gaps (NCCL `dev` at a6112fe8)

- **Not logged at INFO:**
  - Direct `ncclCuMemAlloc` calls in `gin/gin_host_proxy.cc`, `mnnvl.cc`, `rma/rma_proxy.cc`,
    `transport/net_ib/gdaki/gin_host_gdaki.cc` and `transport/p2p.cc`.
  - NVLS buffers (`transport/nvls.cc`).
  - Pool allocations in `allocator.cc`.
- **TRACE only:** `CuMem Alloc` (the size after rounding) and every free. These need an `ENABLE_TRACE` build.
- **Missing from `ncclCommMemStats`:**
  - The `cudaMalloc` fallback in `ncclCudaMalloc/Calloc*` when cuMem is disabled.
  - NET transport buffers. `transport/net.cc` calls `ncclP2pAllocateShareableBuffer` without a memory manager
    at lines 692, 965 and 1145 (both `v2.32.3-1` and `dev`). The manager defaults to `nullptr`.
  - These are visible in the log as `Allocated shareable buffer`. In the T4 run below, they show up as a
    negative `tracked_minus_rounded_estimate`. #2620 (open) passes the manager at all three sites. On
    v2.32.6-1 with it applied, `ncclCommMemStats` went from 16 to 20 MiB (2 GPUs, `rc0`).

## First real run (Kaggle, NCCL v2.32.3-1)

The setup was 2× Tesla T4 (PCIe PHB, no NVLink), driver 580.178.04, CUDA 12.8, a single process and 1 MiB
buffers. With 2 GPUs, NCCL used the SHM transport through host memory, so the first collectives allocated no
GPU memory.

| nGpus | NCCL_RUNTIME_CONNECT | requested | rounded estimate | `ncclStatGpuMemTotal` | device used Δ |
|---|---|---|---|---|---|
| 1 | 1 | 63 KiB (194 allocations) | 388 MiB | 388 MiB | 458 MiB |
| 1 | 0 | 512.06 MiB (195) | 900 MiB | 388 MiB | 970 MiB |
| 2 | 1 | 26 KiB per GPU (8) | 16 MiB | 16 MiB | 86 MiB |
| 2 | 0 | 4.02 MiB per GPU (9) | 20 MiB | 16 MiB | 90 MiB |

- **Rounding.** Excluding shareable buffers, the logged requests rounded to 2 MiB match `ncclStatGpuMemTotal`
  to the byte in every run.
- **Per-channel metadata.** With 1 GPU, 64 channels × 3 per-channel arrays in `channel.cc:44/48/60` request
  40 KiB in total but take 384 MiB.
- **`rc0` vs `rc1`.** The device-wide difference is exactly the untracked shareable buffer: 512 MiB with
  1 GPU, 4 MiB with 2.
- **Unexplained.** About 70 MiB of device-wide difference appears in every run. It isn't explained by the
  logs or `ncclCommMemStats`.
- **Excluded from GPU totals:** host allocations (`Ib Alloc`, `Host Calloc`, `CUMEM Host Alloc`, ...).

## Tests

```sh
python3 -m unittest discover contrib/nccl_memacct -v
```

The tests need no GPU. Most fixtures are built from the format strings in `src/debug.cc` and
`src/include/alloc.h`. `test_data/` holds trimmed, anonymized excerpts of the real Kaggle T4 logs above.
