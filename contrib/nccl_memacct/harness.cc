/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Single-process memory snapshot driver for nccl_memacct.py.
//
// Creates one communicator per visible GPU with ncclCommInitAll, then runs the first AllReduce,
// ReduceScatter and send/recv alltoall. Before each stage it prints a "NCCL_MEMACCT begin=<stage>"
// marker to stdout (the default NCCL log stream), and after each stage it synchronizes every device
// and records cudaMemGetInfo and ncclCommMemStats. Snapshots are taken after synchronization; NCCL
// may still have asynchronous cleanup in flight, so they are not guaranteed settled states.
//
// Usage: NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC ./harness [nGpus] [bytes] [stats.json] > nccl.log

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <vector>

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = (cmd);                                                                  \
    if (e != cudaSuccess) {                                                                 \
      fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                                              \
    }                                                                                       \
  } while (0)

#define NCCLCHECK(cmd)                                                                      \
  do {                                                                                      \
    ncclResult_t r = (cmd);                                                                 \
    if (r != ncclSuccess) {                                                                 \
      fprintf(stderr, "%s:%d NCCL error: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      exit(1);                                                                              \
    }                                                                                       \
  } while (0)

static const char* envKeys[] = {"NCCL_RUNTIME_CONNECT", "NCCL_CUMEM_ENABLE", "NCCL_NVLS_ENABLE",
                                "NCCL_P2P_DISABLE",     "NCCL_SHM_DISABLE",  "NCCL_PROTO",
                                "NCCL_ALGO",            "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"};

struct Ctx {
  int nDev;
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;
  std::vector<float*> send, recv;
  size_t count;
  std::string json;
};

static void begin(const char* stage) {
  printf("NCCL_MEMACCT begin=%s pid=%d\n", stage, (int)getpid());
  fflush(stdout);
}

static void snapshot(Ctx& c, const char* stage) {
  for (int d = 0; d < c.nDev; d++) {
    CUDACHECK(cudaSetDevice(d));
    if (c.streams[d]) CUDACHECK(cudaStreamSynchronize(c.streams[d]));
    CUDACHECK(cudaDeviceSynchronize());
    size_t freeB, totalB;
    CUDACHECK(cudaMemGetInfo(&freeB, &totalB));
    char buf[512];
    snprintf(buf, sizeof(buf), "%s    {\"pid\": %d, \"rank\": %d, \"nranks\": %d, \"dev\": %d, \"stage\": \"%s\", "
             "\"cuda_free\": %zu, \"cuda_total\": %zu, \"nccl_stats\": ",
             c.json.empty() ? "" : ",\n", (int)getpid(), d, c.nDev, d, stage, freeB, totalB);
    c.json += buf;
    uint64_t total = 0, persist = 0, suspend = 0;
    ncclResult_t r = c.comms[d] ? ncclCommMemStats(c.comms[d], ncclStatGpuMemTotal, &total) : ncclInvalidUsage;
    if (r == ncclSuccess) {
      NCCLCHECK(ncclCommMemStats(c.comms[d], ncclStatGpuMemPersist, &persist));
      NCCLCHECK(ncclCommMemStats(c.comms[d], ncclStatGpuMemSuspend, &suspend));
      snprintf(buf, sizeof(buf), "{\"total\": %llu, \"persist\": %llu, \"suspend\": %llu}}",
               (unsigned long long)total, (unsigned long long)persist, (unsigned long long)suspend);
    } else {
      // Memory manager disabled (ncclInvalidUsage) or no communicator yet: unavailable, not zero.
      snprintf(buf, sizeof(buf), "\"unavailable\"}");
    }
    c.json += buf;
  }
}

int main(int argc, char** argv) {
  Ctx c;
  CUDACHECK(cudaGetDeviceCount(&c.nDev));
  if (argc > 1) c.nDev = atoi(argv[1]);
  size_t bytes = argc > 2 ? strtoull(argv[2], nullptr, 0) : (1 << 20);
  const char* statsPath = argc > 3 ? argv[3] : "stats.json";
  c.count = bytes / sizeof(float) / c.nDev * c.nDev;  // divisible by nDev for ReduceScatter and alltoall
  c.comms.assign(c.nDev, nullptr);
  c.streams.assign(c.nDev, nullptr);
  c.send.assign(c.nDev, nullptr);
  c.recv.assign(c.nDev, nullptr);

  // Create contexts and user buffers first so they are not attributed to NCCL.
  for (int d = 0; d < c.nDev; d++) {
    CUDACHECK(cudaSetDevice(d));
    CUDACHECK(cudaFree(0));
    CUDACHECK(cudaStreamCreate(&c.streams[d]));
    CUDACHECK(cudaMalloc(&c.send[d], c.count * sizeof(float)));
    CUDACHECK(cudaMalloc(&c.recv[d], c.count * sizeof(float)));
  }
  snapshot(c, "before_init");

  begin("init");
  NCCLCHECK(ncclCommInitAll(c.comms.data(), c.nDev, nullptr));
  snapshot(c, "init");

  begin("allreduce");
  NCCLCHECK(ncclGroupStart());
  for (int d = 0; d < c.nDev; d++)
    NCCLCHECK(ncclAllReduce(c.send[d], c.recv[d], c.count, ncclFloat, ncclSum, c.comms[d], c.streams[d]));
  NCCLCHECK(ncclGroupEnd());
  snapshot(c, "allreduce");

  begin("reducescatter");
  NCCLCHECK(ncclGroupStart());
  for (int d = 0; d < c.nDev; d++)
    NCCLCHECK(ncclReduceScatter(c.send[d], c.recv[d], c.count / c.nDev, ncclFloat, ncclSum, c.comms[d],
                                c.streams[d]));
  NCCLCHECK(ncclGroupEnd());
  snapshot(c, "reducescatter");

  begin("sendrecv_alltoall");
  size_t chunk = c.count / c.nDev;
  NCCLCHECK(ncclGroupStart());
  for (int d = 0; d < c.nDev; d++) {
    for (int p = 0; p < c.nDev; p++) {
      NCCLCHECK(ncclSend(c.send[d] + p * chunk, chunk, ncclFloat, p, c.comms[d], c.streams[d]));
      NCCLCHECK(ncclRecv(c.recv[d] + p * chunk, chunk, ncclFloat, p, c.comms[d], c.streams[d]));
    }
  }
  NCCLCHECK(ncclGroupEnd());
  snapshot(c, "sendrecv_alltoall");

  FILE* f = fopen(statsPath, "w");
  if (f == nullptr) {
    perror(statsPath);
    return 1;
  }
  fprintf(f, "{\n  \"env\": {");
  for (size_t i = 0; i < sizeof(envKeys) / sizeof(envKeys[0]); i++) {
    const char* v = getenv(envKeys[i]);
    fprintf(f, "%s\"%s\": %s%s%s", i ? ", " : "", envKeys[i], v ? "\"" : "", v ? v : "null", v ? "\"" : "");
  }
  fprintf(f, "},\n  \"bytes\": %zu,\n  \"snapshots\": [\n%s\n  ]\n}\n", bytes, c.json.c_str());
  fclose(f);

  for (int d = 0; d < c.nDev; d++) {
    NCCLCHECK(ncclCommDestroy(c.comms[d]));
    CUDACHECK(cudaSetDevice(d));
    CUDACHECK(cudaFree(c.send[d]));
    CUDACHECK(cudaFree(c.recv[d]));
    CUDACHECK(cudaStreamDestroy(c.streams[d]));
  }
  return 0;
}
