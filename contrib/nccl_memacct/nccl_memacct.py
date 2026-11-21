#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report NCCL GPU allocation requests by allocation site and compare them with ncclCommMemStats.

Input is an NCCL log captured with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC, optionally interleaved with
stage markers and a stats.json written by harness.cc or memtrace. Output is observed allocation requests,
not a complete breakdown of live GPU memory: INFO logs miss some allocation paths and never show frees.
See README.md for what can and cannot be compared.
"""

import argparse
import collections
import json
import re
import sys

# GPU allocation messages emitted at INFO level under NCCL_ALLOC (src/include/alloc.h, src/enqueue/enqueue.cc).
GPU_KINDS = (
  "Cuda Alloc",
  "Cuda Calloc",
  "Cuda CallocAsync",
  "Persistent cudaMalloc work buf",
  "Persistent cudaMallocAsync work buf",
)
# Host allocation messages that share the same line shape; recognized so they are not counted as GPU memory.
HOST_KINDS = (
  "Ib Alloc",
  "CUMEM Host Alloc",
  "Cuda Host Alloc",
  "Host Calloc",
  "Persistent host work buf",
)

_KIND_RE = "|".join(re.escape(k) for k in sorted(GPU_KINDS + HOST_KINDS, key=len, reverse=True))
# Line layout from src/debug.cc: "[timestamp ]host:pid:tid [dev] NCCL INFO " followed by the
# INFO_LOC_FN prefix "file:line (func) " from src/include/debug.h.
_ALLOC_RE = re.compile(
  r"(?P<host>\S+):(?P<pid>\d+):(?P<tid>\d+) \[(?P<dev>-?\d+)\] NCCL INFO "
  r"(?P<file>\S+):(?P<line>\d+) \((?P<func>.+?)\) "
  r"(?P<kind>" + _KIND_RE + r") Size (?P<size>-?\d+) pointer (?P<ptr>\S+)"
  r"(?: memType (?P<memtype>\d+))?"
)
# ncclP2pAllocateShareableBuffer (src/transport/p2p.cc) logs pointer before size. With cuMem it is the only
# INFO line for the buffer; without cuMem the same buffer was already logged as "Cuda Calloc".
SHAREABLE_KIND = "Allocated shareable buffer"
_SHAREABLE_RE = re.compile(
  r"(?P<host>\S+):(?P<pid>\d+):(?P<tid>\d+) \[(?P<dev>-?\d+)\] NCCL INFO "
  r"(?P<file>\S+):(?P<line>\d+) \((?P<func>.+?)\) "
  r"(?P<kind>" + SHAREABLE_KIND + r") (?P<ptr>\S+) size (?P<size>-?\d+)"
)
# Stage marker printed by harness.cc before the work of each stage.
_STAGE_RE = re.compile(r"NCCL_MEMACCT begin=(?P<stage>\S+) pid=(?P<pid>\d+)")

_NULL_PTRS = ("(nil)", "0", "0x0", "(null)", "0000000000000000")

MiB = 1 << 20


class AllocRecord(collections.namedtuple(
    "AllocRecord", "host pid tid dev file line func kind size ptr memtype stage is_gpu ok possible_reuse")):
  """One allocation log line.

  ok is False for failed or zero-size requests, which are logged but hold no memory, and for a shareable-buffer
  line that repeats the pointer of the allocation line just before it (same buffer, already counted).
  """

  @property
  def site(self):
    return "%s:%d (%s)" % (self.file, self.line, self.func)


def parse_log(lines):
  """Parse NCCL ALLOC log lines into AllocRecords, in log order.

  Every recognized line is kept. A pointer that repeats for the same (host, pid, dev) is flagged
  possible_reuse: it suggests an intervening release, but missing log lines make that uncertain.
  """
  records = []
  stage = {}  # pid -> current stage
  live = set()  # (host, pid, dev, ptr) seen so far
  last_ptr = {}  # (host, pid, dev) -> pointer of the most recent GPU allocation line
  for raw in lines:
    m = _STAGE_RE.search(raw)
    if m:
      stage[int(m.group("pid"))] = m.group("stage")
      continue
    m = _ALLOC_RE.search(raw) or _SHAREABLE_RE.search(raw)
    if not m:
      continue
    pid, dev, size, ptr = int(m.group("pid")), int(m.group("dev")), int(m.group("size")), m.group("ptr")
    kind = m.group("kind")
    ok = size > 0 and ptr.lower() not in _NULL_PTRS
    loc = (m.group("host"), pid, dev)
    key = loc + (ptr,)
    if kind == SHAREABLE_KIND and last_ptr.get(loc) == ptr:
      ok = False  # cudaMalloc path: same buffer as the "Cuda Calloc" line just logged
      reuse = False
    else:
      reuse = ok and key in live
      if ok:
        live.add(key)
    if kind in GPU_KINDS or kind == SHAREABLE_KIND:
      last_ptr[loc] = ptr
    memtype = m.groupdict().get("memtype")
    records.append(AllocRecord(
      host=m.group("host"), pid=pid, tid=int(m.group("tid")), dev=dev,
      file=m.group("file"), line=int(m.group("line")), func=m.group("func"),
      kind=kind, size=size, ptr=ptr, memtype=int(memtype) if memtype is not None else None,
      stage=stage.get(pid), is_gpu=kind in GPU_KINDS or kind == SHAREABLE_KIND, ok=ok, possible_reuse=reuse))
  return records


def round_up(size, granularity):
  return -(-size // granularity) * granularity


class SiteTotal(object):
  __slots__ = ("count", "requested", "rounded_estimate", "possible_reuse")

  def __init__(self):
    self.count = 0
    self.requested = 0
    self.rounded_estimate = 0
    self.possible_reuse = 0

  @property
  def rounding_estimate(self):
    return self.rounded_estimate - self.requested


def aggregate(records, granularity=None, by="site"):
  """Sum observed GPU allocation requests per (pid, dev, key), key being the allocation site or source file.

  rounded_estimate rounds each request up to granularity. It is only an estimate, and only
  meaningful when cuMem allocations are in use. Without granularity it equals requested.
  """
  totals = collections.defaultdict(SiteTotal)
  for r in records:
    if not (r.is_gpu and r.ok):
      continue
    key = r.site if by == "site" else r.file
    t = totals[(r.pid, r.dev, key)]
    t.count += 1
    t.requested += r.size
    t.rounded_estimate += round_up(r.size, granularity) if granularity else r.size
    t.possible_reuse += r.possible_reuse
  return dict(totals)


def _stats_total(snapshot):
  stats = snapshot.get("nccl_stats")
  if not isinstance(stats, dict) or "total" not in stats:
    return None  # absent or "unavailable"; never treated as 0
  return stats["total"]


def group_snapshots(snapshots):
  """Collapse per-rank snapshots into one per (pid, dev, stage), keeping first-seen order.

  harness.cc writes one snapshot per device. memtrace writes one per communicator rank, and ranks on the
  same device repeat the same device-wide cuda_free. tracked_total sums ncclStatGpuMemTotal over the ranks
  of a device, skipping a rank whose stats_shared_with names another rank: communicators that share a
  memory manager report the same allocations, so they are counted once (see #2582). tracked_total is None
  when any counted rank has no stats; a missing value is never treated as 0.
  """
  groups = collections.OrderedDict()
  for s in snapshots:
    key = (s["pid"], s["dev"], s["stage"])
    g = groups.get(key)
    if g is None:
      g = groups[key] = {"pid": s["pid"], "dev": s["dev"], "stage": s["stage"], "spec": s.get("spec"),
                         "cuda_free": s.get("cuda_free"), "ranks": [], "tracked_total": 0, "_missing": False}
    g["ranks"].append(s.get("rank"))
    shared = s.get("stats_shared_with")
    if shared is not None and shared >= 0:
      continue
    total = _stats_total(s)
    if total is None:
      g["_missing"] = True
    else:
      g["tracked_total"] += total
  out = []
  for g in groups.values():
    if g.pop("_missing"):
      g["tracked_total"] = None
    out.append(g)
  return out


def step_deltas(snapshots):
  """Per-step changes for each (pid, dev): device-wide used memory and ncclCommMemStats totals.

  device_used_step and tracked_step are the change since the previous stage of the same device, so a
  step that reuses buffers allocated by an earlier step shows 0 (memory is not additive), and a step
  that needs new connections shows its own cost. *_cum is the change since the first stage.
  """
  rows = []
  first, prev = {}, {}
  for g in group_snapshots(snapshots):
    loc = (g["pid"], g["dev"])
    free, tracked = g["cuda_free"], g["tracked_total"]
    base = first.setdefault(loc, free)
    p = prev.get(loc)
    rows.append({
      "pid": g["pid"], "dev": g["dev"], "stage": g["stage"], "spec": g["spec"], "ranks": len(g["ranks"]),
      "device_used_step": None if p is None or p["free"] is None or free is None else p["free"] - free,
      "device_used_cum": None if base is None or free is None else base - free,
      "tracked_total": tracked,
      "tracked_step": None if p is None or p["tracked"] is None or tracked is None else tracked - p["tracked"],
    })
    prev[loc] = {"free": free, "tracked": tracked}
  return rows


def reconcile(records, snapshots, granularity=None):
  """Compare cumulative observed requests with ncclCommMemStats and cudaMemGetInfo at each snapshot.

  Returns one row per (pid, dev, stage) snapshot. Gaps are signed and never expected to be zero:
  - tracked_minus_observed: ncclStatGpuMemTotal minus observed requests (or their rounding estimate).
    The two cover different allocation paths.
  - device_minus_tracked: device-wide used-memory change since the first snapshot minus
    ncclStatGpuMemTotal. This is an unexplained device-wide difference, not untracked NCCL memory.
  """
  groups = group_snapshots(snapshots)
  order = []
  for s in groups:
    if s["stage"] not in order:
      order.append(s["stage"])
  baseline_free = {}
  rows = []
  for s in groups:
    pid, dev, stage = s["pid"], s["dev"], s["stage"]
    baseline_free.setdefault((pid, dev), s.get("cuda_free"))
    upto = set(order[:order.index(stage) + 1])
    observed = rounded = 0
    for r in records:
      if r.is_gpu and r.ok and r.pid == pid and r.dev == dev and (r.stage is None or r.stage in upto):
        observed += r.size
        rounded += round_up(r.size, granularity) if granularity else r.size
    tracked = s["tracked_total"]
    base = baseline_free[(pid, dev)]
    device_used = base - s["cuda_free"] if base is not None and s.get("cuda_free") is not None else None
    rows.append({
      "pid": pid, "dev": dev, "stage": stage,
      "observed_requested": observed,
      "observed_rounded_estimate": rounded if granularity else None,
      "tracked_total": tracked,
      "tracked_minus_observed": None if tracked is None else tracked - observed,
      "tracked_minus_rounded_estimate": None if tracked is None or not granularity else tracked - rounded,
      "device_used_delta": device_used,
      "device_minus_tracked": None if device_used is None or tracked is None else device_used - tracked,
    })
  return rows


def parse_size(text):
  m = re.fullmatch(r"\s*(\d+)\s*([KMG]i?B?)?\s*", text, re.IGNORECASE)
  if not m:
    raise ValueError("bad size: %r" % text)
  unit = (m.group(2) or "").upper()[:1]
  return int(m.group(1)) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[unit]


def _fmt(v):
  if v is None:
    return "n/a"
  return "%d (%.2f MiB)" % (v, v / MiB)


def format_sites(totals, fmt="md"):
  rows = sorted(totals.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1].requested))
  header = ["pid", "dev", "allocation site", "count", "requested", "rounded estimate", "rounding estimate",
            "possible reuse"]
  out = []
  if fmt == "csv":
    out.append(",".join(header))
    for (pid, dev, key), t in rows:
      out.append('%d,%d,"%s",%d,%d,%d,%d,%d' % (pid, dev, key, t.count, t.requested, t.rounded_estimate,
                                               t.rounding_estimate, t.possible_reuse))
  else:
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for (pid, dev, key), t in rows:
      out.append("| %d | %d | `%s` | %d | %s | %s | %s | %d |" % (
        pid, dev, key, t.count, _fmt(t.requested), _fmt(t.rounded_estimate), _fmt(t.rounding_estimate),
        t.possible_reuse))
  return "\n".join(out)


def format_reconcile(rows, fmt="md"):
  cols = ["pid", "dev", "stage", "observed_requested", "observed_rounded_estimate", "tracked_total",
          "tracked_minus_observed", "tracked_minus_rounded_estimate", "device_used_delta", "device_minus_tracked"]
  out = []
  if fmt == "csv":
    out.append(",".join(cols))
    for r in rows:
      out.append(",".join("" if r[c] is None else str(r[c]) for c in cols))
  else:
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "---|" * len(cols))
    for r in rows:
      out.append("| " + " | ".join(str(r[c]) if c in ("pid", "dev", "stage") else _fmt(r[c]) for c in cols) + " |")
  return "\n".join(out)


def _fmt_mib(v, signed=False):
  if v is None:
    return "n/a"
  return ("%+.2f" if signed else "%.2f") % (v / MiB)


def format_steps(rows, fmt="md"):
  cols = ["pid", "dev", "stage", "spec", "ranks", "device_used_step", "device_used_cum", "tracked_total",
          "tracked_step"]
  out = []
  if fmt == "csv":
    out.append(",".join(cols))
    for r in rows:
      out.append(",".join("" if r[c] is None else ('"%s"' % r[c] if c in ("stage", "spec") else str(r[c]))
                          for c in cols))
  else:
    out.append("| " + " | ".join(cols[:5] + [c + " (MiB)" for c in cols[5:]]) + " |")
    out.append("|" + "---|" * len(cols))
    for r in rows:
      out.append("| %d | %d | %s | %s | %d | %s | %s | %s | %s |" % (
        r["pid"], r["dev"], r["stage"], "" if r["spec"] is None else "`%s`" % r["spec"], r["ranks"],
        _fmt_mib(r["device_used_step"], True), _fmt_mib(r["device_used_cum"], True),
        _fmt_mib(r["tracked_total"]), _fmt_mib(r["tracked_step"], True)))
  return "\n".join(out)


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("log", help="NCCL log captured with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC")
  p.add_argument("--stats", help="stats.json written by harness.cc or memtrace")
  p.add_argument("--steps", action="store_true",
                 help="also print per-step deltas (default when the stats file comes from memtrace)")
  p.add_argument("--granularity", type=parse_size, help="cuMem granularity for rounding estimates, e.g. 2MiB")
  p.add_argument("--by", choices=("site", "file"), default="site", help="group by allocation site or source file")
  p.add_argument("--format", choices=("md", "csv"), default="md")
  args = p.parse_args(argv)

  with open(args.log, errors="replace") as f:
    records = parse_log(f)
  print(format_sites(aggregate(records, args.granularity, args.by), args.format))
  skipped = sum(1 for r in records if r.is_gpu and not r.ok)
  host = sum(1 for r in records if not r.is_gpu)
  print("\n%d GPU allocation lines, %d failed or zero-size (not counted), %d host allocation lines (excluded)."
        % (sum(1 for r in records if r.is_gpu), skipped, host))
  if args.stats:
    with open(args.stats) as f:
      stats = json.load(f)
    snapshots = stats["snapshots"]
    print()
    print(format_reconcile(reconcile(records, snapshots, args.granularity), args.format))
    if args.steps or "steps" in stats:
      print()
      print(format_steps(step_deltas(snapshots), args.format))
  return 0


if __name__ == "__main__":
  sys.exit(main())
