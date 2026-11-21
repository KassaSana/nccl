/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// memtrace: run a sequence of NCCL operations and record GPU memory after each one.
//
// One process drives nGpus x ranksPerGpu ranks of one communicator. After communicator init and after
// every step it synchronizes all streams and records:
//   - cudaMemGetInfo once per device (device-wide; several ranks can share a device), and
//   - ncclCommMemStats (total, persist, suspend) once per communicator.
// Steps run back to back on the same communicator, so the per-step delta shows whether an operation
// allocates new memory or reuses what earlier steps allocated.
//
// Ops mirror nccl-tests (src/<op>.cu): the same runColl signature, the same getCollByteCount rules
// (size = per-rank buffer bytes, as in -b/-e), out-of-place, root 0. See README.md.
//
// stdout carries "NCCL_MEMACCT begin=<stage>" markers interleaved with the NCCL log, so
// nccl_memacct.py can attribute NCCL_DEBUG_SUBSYS=ALLOC lines to steps. The per-step table goes to stderr.
//
// Usage: memtrace [-g nGpus] [-r ranksPerGpu] [-o ops | -f opsfile] [-d dtype] [-j out.json]
//   e.g. NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC ./memtrace -g 1 -r 2 -o sendrecv:1M,allreduce:1M,allreduce:64M > nccl.log

#include <cuda_runtime.h>
#include <nccl.h>
#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "memtrace: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(2);
}

// ------------------------------------------------------------------------------------------- ops
// Same signatures as nccl-tests' struct testColl (src/common.h), minus initData/getBw, which this
// tool does not need. Counts are in elements.
typedef void (*GetCollByteCount_t)(size_t* sendcount, size_t* recvcount, size_t* paramcount,
                                   size_t* sendInplaceOffset, size_t* recvInplaceOffset, size_t count,
                                   size_t eltSize, int nranks);
typedef ncclResult_t (*RunColl_t)(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset,
                                  size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm,
                                  cudaStream_t stream);

// count, count: all_reduce.cu, broadcast.cu, reduce.cu, sendrecv.cu
static void FullGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount, size_t* sendInplaceOffset,
                                 size_t* recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  (void)eltSize;
  (void)nranks;
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = count;
}
// reduce_scatter.cu
static void ReduceScatterGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount,
                                          size_t* sendInplaceOffset, size_t* recvInplaceOffset, size_t count,
                                          size_t eltSize, int nranks) {
  size_t base = (count / nranks) & ~(16 / eltSize - 1);
  *sendcount = base * nranks;
  *recvcount = base;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = base;
  *paramcount = base;
}
// all_gather.cu
static void AllGatherGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount,
                                      size_t* sendInplaceOffset, size_t* recvInplaceOffset, size_t count,
                                      size_t eltSize, int nranks) {
  size_t base = (count / nranks) & ~(16 / eltSize - 1);
  *sendcount = base;
  *recvcount = base * nranks;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}
// alltoall.cu
static void AlltoAllGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount,
                                     size_t* sendInplaceOffset, size_t* recvInplaceOffset, size_t count,
                                     size_t eltSize, int nranks) {
  *paramcount = (count / nranks) & ~(16 / eltSize - 1);
  *sendcount = nranks * (*paramcount);
  *recvcount = *sendcount;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
}

static size_t eltSizeOf(ncclDataType_t t) {
  switch (t) {
  case ncclInt8: case ncclUint8: return 1;
  case ncclFloat16: case ncclBfloat16: return 2;
  case ncclInt32: case ncclUint32: case ncclFloat32: return 4;
  default: return 8;
  }
}

static ncclResult_t AllReduceRun(void* s, size_t so, void* r, size_t ro, size_t count, ncclDataType_t type,
                                 ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  (void)root;
  return ncclAllReduce((char*)s + so, (char*)r + ro, count, type, op, comm, stream);
}
static ncclResult_t ReduceScatterRun(void* s, size_t so, void* r, size_t ro, size_t count, ncclDataType_t type,
                                     ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  (void)root;
  return ncclReduceScatter((char*)s + so, (char*)r + ro, count, type, op, comm, stream);
}
static ncclResult_t AllGatherRun(void* s, size_t so, void* r, size_t ro, size_t count, ncclDataType_t type,
                                 ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  (void)op;
  (void)root;
  return ncclAllGather((char*)s + so, (char*)r + ro, count, type, comm, stream);
}
static ncclResult_t BroadcastRun(void* s, size_t so, void* r, size_t ro, size_t count, ncclDataType_t type,
                                 ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  (void)op;
  return ncclBroadcast((char*)s + so, (char*)r + ro, count, type, root, comm, stream);
}
static ncclResult_t ReduceRun(void* s, size_t so, void* r, size_t ro, size_t count, ncclDataType_t type,
                              ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  return ncclReduce((char*)s + so, (char*)r + ro, count, type, op, root, comm, stream);
}
// sendrecv.cu: ring, send to rank+1 and receive from rank-1.
static ncclResult_t SendRecvRun(void* s, size_t so, void* r, size_t ro, size_t count, ncclDataType_t type,
                                ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  (void)op;
  (void)root;
  int nRanks, rank;
  ncclResult_t res;
  if ((res = ncclCommCount(comm, &nRanks)) != ncclSuccess) return res;
  if ((res = ncclCommUserRank(comm, &rank)) != ncclSuccess) return res;
  if ((res = ncclGroupStart()) != ncclSuccess) return res;
  if ((res = ncclSend((char*)s + so, count, type, (rank + 1) % nRanks, comm, stream)) != ncclSuccess) return res;
  if ((res = ncclRecv((char*)r + ro, count, type, (rank - 1 + nRanks) % nRanks, comm, stream)) != ncclSuccess)
    return res;
  return ncclGroupEnd();
}
// alltoall.cu's send/recv implementation (its fallback before NCCL 2.28), and the "send/recv based
// alltoall" pattern measured in #2527.
static ncclResult_t AlltoAllSendRecvRun(void* s, size_t so, void* r, size_t ro, size_t count, ncclDataType_t type,
                                        ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  (void)op;
  (void)root;
  int nRanks;
  ncclResult_t res;
  if ((res = ncclCommCount(comm, &nRanks)) != ncclSuccess) return res;
  size_t rankOffset = count * eltSizeOf(type);
  if ((res = ncclGroupStart()) != ncclSuccess) return res;
  for (int p = 0; p < nRanks; p++) {
    if ((res = ncclSend((char*)s + so + p * rankOffset, count, type, p, comm, stream)) != ncclSuccess) return res;
    if ((res = ncclRecv((char*)r + ro + p * rankOffset, count, type, p, comm, stream)) != ncclSuccess) return res;
  }
  return ncclGroupEnd();
}
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
// alltoall.cu on NCCL >= 2.28: the native collective.
static ncclResult_t AlltoAllRun(void* s, size_t so, void* r, size_t ro, size_t count, ncclDataType_t type,
                                ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  (void)op;
  (void)root;
  return ncclAlltoAll((char*)s + so, (char*)r + ro, count, type, comm, stream);
}
#endif

struct OpDesc {
  const char* name;
  GetCollByteCount_t getCollByteCount;
  RunColl_t runColl;
};

static const OpDesc opTable[] = {
  {"allreduce", FullGetCollByteCount, AllReduceRun},
  {"reducescatter", ReduceScatterGetCollByteCount, ReduceScatterRun},
  {"allgather", AllGatherGetCollByteCount, AllGatherRun},
  {"broadcast", FullGetCollByteCount, BroadcastRun},
  {"reduce", FullGetCollByteCount, ReduceRun},
  {"sendrecv", FullGetCollByteCount, SendRecvRun},
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  {"alltoall", AlltoAllGetCollByteCount, AlltoAllRun},
#endif
  {"a2a_sendrecv", AlltoAllGetCollByteCount, AlltoAllSendRecvRun},
};

static const OpDesc* findOp(const std::string& name) {
  for (size_t i = 0; i < sizeof(opTable) / sizeof(opTable[0]); i++)
    if (name == opTable[i].name) return &opTable[i];
  return nullptr;
}

// ------------------------------------------------------------------------------------- op spec
// step = op ":" size [":" key=val]* ; steps are separated by ',' or newlines; '#' starts a comment.
struct Step {
  const OpDesc* op;
  std::string spec;   // as written, for the report
  std::string stage;  // unique stage name, e.g. "s03_allreduce_64M"
  size_t bytes;       // per-rank size, nccl-tests -b semantics
  int iters = 1;
  int root = 0;
  size_t sendcount = 0, recvcount = 0, paramcount = 0;  // elements, from getCollByteCount
};

static std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && isspace((unsigned char)s[b])) b++;
  while (e > b && isspace((unsigned char)s[e - 1])) e--;
  return s.substr(b, e - b);
}

static bool parseSize(const std::string& text, size_t* out) {
  std::string t = trim(text);
  if (t.empty() || !isdigit((unsigned char)t[0])) return false;
  char* end = nullptr;
  unsigned long long v = strtoull(t.c_str(), &end, 10);
  std::string unit = end;
  for (auto& c : unit) c = (char)toupper((unsigned char)c);
  size_t mult;
  if (unit == "" || unit == "B") mult = 1;
  else if (unit == "K" || unit == "KB" || unit == "KIB") mult = 1ULL << 10;
  else if (unit == "M" || unit == "MB" || unit == "MIB") mult = 1ULL << 20;
  else if (unit == "G" || unit == "GB" || unit == "GIB") mult = 1ULL << 30;
  else return false;
  *out = (size_t)v * mult;
  return true;
}

static bool parseInt(const std::string& text, int* out) {
  char* end = nullptr;
  long v = strtol(text.c_str(), &end, 10);
  if (text.empty() || *end != '\0' || v < INT_MIN || v > INT_MAX) return false;
  *out = (int)v;
  return true;
}

static std::string fmtSize(size_t b) {
  char buf[32];
  if (b && b % (1ULL << 30) == 0) snprintf(buf, sizeof(buf), "%zuG", b >> 30);
  else if (b && b % (1ULL << 20) == 0) snprintf(buf, sizeof(buf), "%zuM", b >> 20);
  else if (b && b % (1ULL << 10) == 0) snprintf(buf, sizeof(buf), "%zuK", b >> 10);
  else snprintf(buf, sizeof(buf), "%zu", b);
  return buf;
}

static std::vector<std::string> split(const std::string& s, const char* seps) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (strchr(seps, c)) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  out.push_back(cur);
  return out;
}

static std::vector<Step> parseSpec(const std::string& text, ncclDataType_t type, int nRanks) {
  std::vector<Step> steps;
  size_t elt = eltSizeOf(type);
  for (std::string ln : split(text, "\n")) {
    size_t hash = ln.find('#');
    if (hash != std::string::npos) ln = ln.substr(0, hash);
    for (std::string tok : split(ln, ",")) {
      tok = trim(tok);
      if (tok.empty()) continue;
      std::vector<std::string> f = split(tok, ":");
      Step st;
      st.spec = tok;
      st.op = findOp(trim(f[0]));
      if (!st.op) die("unknown op '%s' in step '%s'", trim(f[0]).c_str(), tok.c_str());
      if (f.size() < 2 || !parseSize(f[1], &st.bytes) || st.bytes == 0)
        die("step '%s': expected op:size, e.g. allreduce:1M", tok.c_str());
      for (size_t i = 2; i < f.size(); i++) {
        std::string kv = trim(f[i]);
        size_t eq = kv.find('=');
        std::string k = trim(kv.substr(0, eq)), v = eq == std::string::npos ? "" : trim(kv.substr(eq + 1));
        if (k == "iters") {
          if (!parseInt(v, &st.iters) || st.iters < 1) die("step '%s': iters must be an integer >= 1", tok.c_str());
        } else if (k == "root") {
          if (!parseInt(v, &st.root) || st.root < 0 || st.root >= nRanks)
            die("step '%s': root must be an integer in [0, %d)", tok.c_str(), nRanks);
        } else if (k == "comm" || k == "graph") {
          die("step '%s': '%s' is reserved for a later version (multiple communicators, CUDA graphs)",
              tok.c_str(), k.c_str());
        } else {
          die("step '%s': unknown key '%s' (known: iters, root)", tok.c_str(), k.c_str());
        }
      }
      if (st.bytes % elt) die("step '%s': size is not a multiple of the %zu-byte element", tok.c_str(), elt);
      size_t sio, rio;
      st.op->getCollByteCount(&st.sendcount, &st.recvcount, &st.paramcount, &sio, &rio, st.bytes / elt, elt,
                              nRanks);
      if (st.paramcount == 0)
        die("step '%s': size too small for %d ranks (per-rank chunk rounds down to 0)", tok.c_str(), nRanks);
      char stage[96];
      snprintf(stage, sizeof(stage), "s%02zu_%s_%s", steps.size() + 1, st.op->name, fmtSize(st.bytes).c_str());
      st.stage = stage;
      steps.push_back(st);
    }
  }
  if (steps.empty()) die("no steps given (-o or -f)");
  return steps;
}

static ncclDataType_t parseType(const char* s) {
  struct {
    const char* n;
    ncclDataType_t t;
  } types[] = {{"int8", ncclInt8},     {"uint8", ncclUint8},       {"int32", ncclInt32},
               {"uint32", ncclUint32}, {"float", ncclFloat32},     {"half", ncclFloat16},
               {"bfloat16", ncclBfloat16}, {"double", ncclFloat64}};
  for (auto& t : types)
    if (!strcmp(s, t.n)) return t.t;
  die("unknown dtype '%s'", s);
  return ncclFloat32;
}

// ------------------------------------------------------------------------------- run + record
// One entry per communicator rank. Later versions (split/shrink/shared) append more entries; ranks
// whose communicator shares a memory manager with an earlier entry set statsSharedWith (#2582).
struct CommInfo {
  ncclComm_t comm = nullptr;
  int rank = 0, dev = 0;
  int statsSharedWith = -1;  // index of the CommInfo whose ncclCommMemStats this one duplicates, or -1
  cudaStream_t stream = nullptr;
  void *send = nullptr, *recv = nullptr;
};

struct DevSample {
  size_t freeB = 0, totalB = 0;
};
struct CommSample {
  bool ok = false;
  uint64_t total = 0, persist = 0, suspend = 0;
};

struct Ctx {
  int nGpus = 0, ranksPerGpu = 1, nRanks = 0;
  std::vector<CommInfo> comms;
  std::string json;
  std::vector<DevSample> base, prevDev;
  std::vector<CommSample> prevComm;
  bool havePrev = false;
};

static void begin(const char* stage) {
  printf("NCCL_MEMACCT begin=%s pid=%d\n", stage, (int)getpid());
  fflush(stdout);
}

static void syncAll(Ctx& c) {
  for (auto& ci : c.comms) {
    CUDACHECK(cudaSetDevice(ci.dev));
    CUDACHECK(cudaStreamSynchronize(ci.stream));
  }
  for (int d = 0; d < c.nGpus; d++) {
    CUDACHECK(cudaSetDevice(d));
    CUDACHECK(cudaDeviceSynchronize());
  }
}

static double mib(long long b) { return (double)b / (1 << 20); }

static std::string jsonStr(const std::string& s) {
  std::string o = "\"";
  for (char ch : s) {
    if (ch == '"' || ch == '\\') o += '\\';
    if ((unsigned char)ch < 0x20) continue;
    o += ch;
  }
  return o + "\"";
}

// Records one snapshot per communicator rank, in the schema of harness.cc's stats.json plus step fields.
// cuda_free is device-wide and repeats for ranks on the same device; nccl_memacct.py keeps one per device.
static void snapshot(Ctx& c, int stepIdx, const char* stage, const char* op, size_t bytes, int iters,
                     bool commsLive) {
  syncAll(c);
  std::vector<DevSample> dev(c.nGpus);
  for (int d = 0; d < c.nGpus; d++) {
    CUDACHECK(cudaSetDevice(d));
    CUDACHECK(cudaMemGetInfo(&dev[d].freeB, &dev[d].totalB));
  }
  std::vector<CommSample> cs(c.comms.size());
  for (size_t i = 0; i < c.comms.size(); i++) {
    CommInfo& ci = c.comms[i];
    if (!commsLive || ci.comm == nullptr) continue;
    // ncclInvalidUsage means the memory manager is disabled: unavailable, never 0. Any other error is fatal.
    ncclResult_t res = ncclCommMemStats(ci.comm, ncclStatGpuMemTotal, &cs[i].total);
    if (res == ncclInvalidUsage) continue;
    NCCLCHECK(res);
    NCCLCHECK(ncclCommMemStats(ci.comm, ncclStatGpuMemPersist, &cs[i].persist));
    NCCLCHECK(ncclCommMemStats(ci.comm, ncclStatGpuMemSuspend, &cs[i].suspend));
    cs[i].ok = true;
  }
  if (c.base.empty()) c.base = dev;

  char buf[768];
  std::string specJson = op ? jsonStr(op) : "null";
  for (size_t i = 0; i < c.comms.size(); i++) {
    const CommInfo& ci = c.comms[i];
    // spec is user text of any length, so it is appended outside the fixed-size buffer.
    snprintf(buf, sizeof(buf),
             "%s    {\"pid\": %d, \"rank\": %d, \"nranks\": %d, \"dev\": %d, \"stage\": \"%s\", \"step\": %d, "
             "\"spec\": ",
             c.json.empty() ? "" : ",\n", (int)getpid(), ci.rank, c.nRanks, ci.dev, stage, stepIdx);
    c.json += buf;
    c.json += specJson;
    snprintf(buf, sizeof(buf),
             ", \"bytes\": %zu, \"iters\": %d, \"stats_shared_with\": %d, \"cuda_free\": %zu, \"cuda_total\": %zu, "
             "\"nccl_stats\": ",
             bytes, iters, ci.statsSharedWith, dev[ci.dev].freeB, dev[ci.dev].totalB);
    c.json += buf;
    if (cs[i].ok)
      snprintf(buf, sizeof(buf), "{\"total\": %llu, \"persist\": %llu, \"suspend\": %llu}}",
               (unsigned long long)cs[i].total, (unsigned long long)cs[i].persist,
               (unsigned long long)cs[i].suspend);
    else
      snprintf(buf, sizeof(buf), "\"unavailable\"}");
    c.json += buf;
  }

  // Human-readable rows on stderr: one per device. The stats column sums the communicators on that device,
  // counting a shared memory manager once; per-rank values are in the JSON.
  for (int d = 0; d < c.nGpus; d++) {
    long long usedCum = (long long)c.base[d].freeB - (long long)dev[d].freeB;
    long long usedStep = c.havePrev ? (long long)c.prevDev[d].freeB - (long long)dev[d].freeB : 0;
    bool any = false, allOk = true;
    long long tot = 0, totPrev = 0;
    for (size_t i = 0; i < c.comms.size(); i++) {
      if (c.comms[i].dev != d || c.comms[i].statsSharedWith >= 0) continue;
      if (!cs[i].ok) {
        allOk = false;
        continue;
      }
      any = true;
      tot += (long long)cs[i].total;
      if (c.havePrev && c.prevComm[i].ok) totPrev += (long long)c.prevComm[i].total;
    }
    char stats[64];
    if (any && allOk)
      snprintf(stats, sizeof(stats), "%10.2f %+10.2f", mib(tot), mib(tot - totPrev));
    else
      snprintf(stats, sizeof(stats), "%10s %10s", "n/a", "");
    fprintf(stderr, "%-24s %-24s %3d %+12.2f %+12.2f %s\n", stage, op ? op : "-", d, mib(usedStep), mib(usedCum),
            stats);
  }
  c.prevDev = dev;
  c.prevComm = cs;
  c.havePrev = true;
}

static const char* envKeys[] = {
  "NCCL_RUNTIME_CONNECT", "NCCL_CUMEM_ENABLE",     "NCCL_NVLS_ENABLE",       "NCCL_P2P_DISABLE",
  "NCCL_SHM_DISABLE",     "NCCL_NET_SHARED_BUFFERS", "NCCL_PROTO",            "NCCL_ALGO",
  "NCCL_MIN_NCHANNELS",   "NCCL_MAX_NCHANNELS",    "NCCL_MULTI_RANK_GPU_ENABLE", "NCCL_DISABLE_MEM_MANAGER"};

static void usage() {
  fprintf(stderr,
          "usage: memtrace [-g nGpus] [-r ranksPerGpu] (-o ops | -f opsfile) [-d dtype] [-j out.json]\n"
          "  ops: comma- or newline-separated steps, op:size[:iters=N][:root=N]\n"
          "  op:  allreduce reducescatter allgather broadcast reduce sendrecv a2a_sendrecv"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
          " alltoall"
#endif
          "\n  size: per-rank buffer bytes as in nccl-tests -b, with K/M/G suffixes\n"
          "  e.g. -o sendrecv:1M,allreduce:1M,allreduce:64M,allreduce:1M\n");
  exit(2);
}

int main(int argc, char** argv) {
  Ctx c;
  CUDACHECK(cudaGetDeviceCount(&c.nGpus));
  std::string spec, jsonPath = "memtrace.json";
  ncclDataType_t type = ncclFloat32;
  const char* typeName = "float";
  int opt;
  while ((opt = getopt(argc, argv, "g:r:o:f:d:j:h")) != -1) {
    switch (opt) {
    case 'g': c.nGpus = atoi(optarg); break;
    case 'r': c.ranksPerGpu = atoi(optarg); break;
    case 'o': spec += std::string(optarg) + "\n"; break;
    case 'f': {
      FILE* f = fopen(optarg, "r");
      if (!f) die("cannot open %s", optarg);
      char b[4096];
      size_t n;
      while ((n = fread(b, 1, sizeof(b), f)) > 0) spec.append(b, n);
      fclose(f);
      spec += "\n";
      break;
    }
    case 'd': type = parseType(optarg); typeName = optarg; break;
    case 'j': jsonPath = optarg; break;
    default: usage();
    }
  }
  int nVisible;
  CUDACHECK(cudaGetDeviceCount(&nVisible));
  if (c.nGpus < 1 || c.nGpus > nVisible) die("-g %d: %d device(s) visible", c.nGpus, nVisible);
  if (c.ranksPerGpu < 1) die("-r must be >= 1");
  c.nRanks = c.nGpus * c.ranksPerGpu;
  std::vector<Step> steps = parseSpec(spec, type, c.nRanks);

  // Several ranks on one GPU is gated in NCCL (init.cc, NCCL_MULTI_RANK_GPU_ENABLE). Ranks on the same GPU
  // then use the P2P transport with device buffers, which makes connection memory visible on
  // machines whose GPU pairs would otherwise use SHM (host memory).
  if (c.ranksPerGpu > 1) setenv("NCCL_MULTI_RANK_GPU_ENABLE", "1", 0);

  // User buffers are sized for the largest step and allocated before the baseline, so no step's delta
  // includes them (nccl-tests -M reports them separately as "User-Allocated").
  size_t elt = eltSizeOf(type), maxSend = 0, maxRecv = 0;
  for (auto& s : steps) {
    if (s.sendcount * elt > maxSend) maxSend = s.sendcount * elt;
    if (s.recvcount * elt > maxRecv) maxRecv = s.recvcount * elt;
  }
  c.comms.resize(c.nRanks);
  for (int r = 0; r < c.nRanks; r++) {
    CommInfo& ci = c.comms[r];
    ci.rank = r;
    ci.dev = r / c.ranksPerGpu;
    CUDACHECK(cudaSetDevice(ci.dev));
    CUDACHECK(cudaFree(0));
    CUDACHECK(cudaStreamCreateWithFlags(&ci.stream, cudaStreamNonBlocking));
    CUDACHECK(cudaMalloc(&ci.send, maxSend));
    CUDACHECK(cudaMalloc(&ci.recv, maxRecv));
    CUDACHECK(cudaMemset(ci.send, 0, maxSend));
    CUDACHECK(cudaMemset(ci.recv, 0, maxRecv));
  }

  int version = 0;
  NCCLCHECK(ncclGetVersion(&version));
  fprintf(stderr, "memtrace: NCCL %d, %d GPU(s) x %d rank(s) = %d ranks, dtype %s, %zu step(s)\n", version, c.nGpus,
          c.ranksPerGpu, c.nRanks, typeName, steps.size());
  fprintf(stderr, "%-24s %-24s %3s %12s %12s %10s %10s\n", "stage", "op", "dev", "dev_d_step", "dev_d_cum",
          "stats_MiB", "stats_d");
  fprintf(stderr, "%-24s %-24s %3s %12s %12s %10s %10s\n", "", "", "", "MiB", "MiB", "(sum)", "MiB");
  snapshot(c, 0, "before_init", nullptr, 0, 0, false);

  begin("init");
  ncclUniqueId id;
  NCCLCHECK(ncclGetUniqueId(&id));
  NCCLCHECK(ncclGroupStart());
  for (auto& ci : c.comms) {
    CUDACHECK(cudaSetDevice(ci.dev));
    NCCLCHECK(ncclCommInitRank(&ci.comm, c.nRanks, id, ci.rank));
  }
  NCCLCHECK(ncclGroupEnd());
  snapshot(c, 0, "init", "init", 0, 0, true);

  for (size_t i = 0; i < steps.size(); i++) {
    const Step& s = steps[i];
    begin(s.stage.c_str());
    for (int it = 0; it < s.iters; it++) {
      // One group across all ranks driven by this thread, as nccl-tests does for -g > 1.
      NCCLCHECK(ncclGroupStart());
      for (auto& ci : c.comms) {
        CUDACHECK(cudaSetDevice(ci.dev));
        NCCLCHECK(s.op->runColl(ci.send, 0, ci.recv, 0, s.paramcount, type, ncclSum, s.root, ci.comm, ci.stream));
      }
      NCCLCHECK(ncclGroupEnd());
    }
    snapshot(c, (int)i + 1, s.stage.c_str(), s.spec.c_str(), s.bytes, s.iters, true);
  }

  // Teardown: device memory should return to the before_init level once every communicator is destroyed.
  begin("destroy");
  for (auto& ci : c.comms) {
    CUDACHECK(cudaSetDevice(ci.dev));
    NCCLCHECK(ncclCommDestroy(ci.comm));
    ci.comm = nullptr;
  }
  snapshot(c, (int)steps.size() + 1, "destroy", "destroy", 0, 0, false);

  FILE* f = fopen(jsonPath.c_str(), "w");
  if (f == nullptr) {
    perror(jsonPath.c_str());
    return 1;
  }
  fprintf(f, "{\n  \"tool\": \"memtrace\",\n  \"nccl_version\": %d,\n  \"ngpus\": %d,\n  \"ranks_per_gpu\": %d,\n"
          "  \"dtype\": %s,\n  \"env\": {", version, c.nGpus, c.ranksPerGpu, jsonStr(typeName).c_str());
  for (size_t i = 0; i < sizeof(envKeys) / sizeof(envKeys[0]); i++) {
    const char* v = getenv(envKeys[i]);
    fprintf(f, "%s\"%s\": %s", i ? ", " : "", envKeys[i], v ? jsonStr(v).c_str() : "null");
  }
  fprintf(f, "},\n  \"steps\": [");
  for (size_t i = 0; i < steps.size(); i++) {
    const Step& s = steps[i];
    fprintf(f, "%s\n    {\"step\": %zu, \"stage\": %s, \"spec\": %s, \"op\": %s, \"bytes\": %zu, \"iters\": %d, "
            "\"root\": %d, \"sendcount\": %zu, \"recvcount\": %zu, \"paramcount\": %zu}",
            i ? "," : "", i + 1, jsonStr(s.stage).c_str(), jsonStr(s.spec).c_str(), jsonStr(s.op->name).c_str(),
            s.bytes, s.iters, s.root, s.sendcount, s.recvcount, s.paramcount);
  }
  fprintf(f, "\n  ],\n  \"user_buffer_bytes\": {\"send\": %zu, \"recv\": %zu},\n  \"snapshots\": [\n%s\n  ]\n}\n",
          maxSend, maxRecv, c.json.c_str());
  fclose(f);

  for (auto& ci : c.comms) {
    CUDACHECK(cudaSetDevice(ci.dev));
    CUDACHECK(cudaFree(ci.send));
    CUDACHECK(cudaFree(ci.recv));
    CUDACHECK(cudaStreamDestroy(ci.stream));
  }
  return 0;
}
