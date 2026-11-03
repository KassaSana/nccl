# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for nccl_memacct.

Most fixture lines are built from the format strings in src/debug.cc and src/include/alloc.h.
RealLogTest uses trimmed, anonymized excerpts of real NCCL v2.32.3-1 logs in test_data/.
"""

import collections
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nccl_memacct as m  # noqa: E402

MiB = 1 << 20


def line(kind, size, ptr, file="transport/p2p.cc", lineno=500, func="p2pSendSetup", memtype=0,
         host="node1", pid=100, tid=101, dev=0, ts=""):
  # "%s:%d:%d " + "[%d] NCCL INFO %s" with INFO_LOC_FN's "%s:%d (%s) " prefix.
  tail = " memType %d" % memtype if kind.startswith("Cuda ") and "Host" not in kind else ""
  return "%s%s:%d:%d [%d] NCCL INFO %s:%d (%s) %s Size %d pointer %s%s\n" % (
    ts, host, pid, tid, dev, file, lineno, func, kind, size, ptr, tail)


def marker(stage, pid=100):
  return "NCCL_MEMACCT begin=%s pid=%d\n" % (stage, pid)


class ParseTest(unittest.TestCase):
  def test_each_gpu_kind(self):
    lines = [line("Cuda Alloc", 10, "0x7f0000000000"),
             line("Cuda Calloc", 20, "0x7f0000200000"),
             line("Cuda CallocAsync", 30, "0x7f0000400000"),
             line("Persistent cudaMalloc work buf", 40, "0x7f0000600000", file="enqueue/enqueue.cc"),
             line("Persistent cudaMallocAsync work buf", 50, "0x7f0000800000", file="enqueue/enqueue.cc")]
    recs = m.parse_log(lines)
    self.assertEqual([r.kind for r in recs], ["Cuda Alloc", "Cuda Calloc", "Cuda CallocAsync",
                                              "Persistent cudaMalloc work buf",
                                              "Persistent cudaMallocAsync work buf"])
    self.assertEqual([r.size for r in recs], [10, 20, 30, 40, 50])
    self.assertTrue(all(r.is_gpu and r.ok for r in recs))
    self.assertEqual(recs[0].memtype, 0)
    self.assertIsNone(recs[3].memtype)

  def test_timestamp_prefix_and_site(self):
    r, = m.parse_log([line("Cuda Calloc", 64, "0x1000", ts="[2026-10-28 10:00:00] ", func="initChannel")])
    self.assertEqual(r.site, "transport/p2p.cc:500 (initChannel)")
    self.assertEqual((r.host, r.pid, r.tid, r.dev), ("node1", 100, 101, 0))

  def test_func_with_parentheses(self):
    r, = m.parse_log([line("Cuda Alloc", 8, "0x1000", func="operator()")])
    self.assertEqual(r.func, "operator()")

  def test_host_kinds_excluded_from_gpu(self):
    recs = m.parse_log([line("Ib Alloc", 4096, "0x5555"), line("Host Calloc", 128, "0x5556"),
                        line("Cuda Host Alloc", 256, "0x5557"), line("Persistent host work buf", 512, "0x5558")])
    self.assertEqual(len(recs), 4)
    self.assertFalse(any(r.is_gpu for r in recs))
    self.assertEqual(m.aggregate(recs), {})

  def test_failed_and_zero_size_not_counted(self):
    recs = m.parse_log([line("Cuda Alloc", 1024, "(nil)"), line("Cuda Calloc", 0, "(nil)"),
                        line("Cuda Calloc", 0, "0x0")])
    self.assertEqual(len(recs), 3)
    self.assertFalse(any(r.ok for r in recs))
    self.assertEqual(m.aggregate(recs), {})

  def test_unrelated_and_malformed_lines_skipped(self):
    recs = m.parse_log(["node1:100:101 [0] NCCL INFO Channel 00/02 : 0 1\n",
                        "node1:100:101 [0] NCCL INFO transport/p2p.cc:500 (f) Cuda Calloc Size abc pointer 0x1\n",
                        "garbage\n", "",
                        "node1:100:101 [0] NCCL INFO MemManager: Initialized for device 0\n"])
    self.assertEqual(recs, [])

  def test_repeated_pointer_kept_and_flagged(self):
    recs = m.parse_log([line("Cuda Calloc", 100, "0xa000"), line("Cuda Calloc", 200, "0xa000")])
    self.assertEqual(len(recs), 2)
    self.assertEqual([r.possible_reuse for r in recs], [False, True])
    t = m.aggregate(recs)[(100, 0, "transport/p2p.cc:500 (p2pSendSetup)")]
    self.assertEqual((t.count, t.requested, t.possible_reuse), (2, 300, 1))

  def test_same_pointer_different_process_or_device_not_reuse(self):
    recs = m.parse_log([line("Cuda Calloc", 1, "0xa000", pid=100), line("Cuda Calloc", 1, "0xa000", pid=200),
                        line("Cuda Calloc", 1, "0xa000", pid=100, dev=1)])
    self.assertFalse(any(r.possible_reuse for r in recs))

  def test_failed_allocation_does_not_mark_pointer(self):
    recs = m.parse_log([line("Cuda Calloc", 10, "(nil)"), line("Cuda Calloc", 10, "(nil)")])
    self.assertFalse(any(r.possible_reuse for r in recs))

  def test_stage_markers_per_process(self):
    recs = m.parse_log([marker("init", 100), line("Cuda Calloc", 1, "0x1", pid=100),
                        marker("init", 200), line("Cuda Calloc", 1, "0x1", pid=200),
                        marker("allreduce", 100), line("Cuda Calloc", 1, "0x2", pid=100),
                        line("Cuda Calloc", 1, "0x2", pid=200)])
    self.assertEqual([(r.pid, r.stage) for r in recs],
                     [(100, "init"), (200, "init"), (100, "allreduce"), (200, "init")])


class AggregateTest(unittest.TestCase):
  def test_groups_by_process_device_and_site(self):
    recs = m.parse_log([line("Cuda Calloc", 100, "0x1", dev=0, lineno=10),
                        line("Cuda Calloc", 50, "0x2", dev=0, lineno=10),
                        line("Cuda Calloc", 70, "0x3", dev=1, lineno=10),
                        line("Cuda Calloc", 30, "0x4", dev=0, lineno=20)])
    t = m.aggregate(recs)
    self.assertEqual(t[(100, 0, "transport/p2p.cc:10 (p2pSendSetup)")].requested, 150)
    self.assertEqual(t[(100, 1, "transport/p2p.cc:10 (p2pSendSetup)")].requested, 70)
    self.assertEqual(t[(100, 0, "transport/p2p.cc:20 (p2pSendSetup)")].requested, 30)
    by_file = m.aggregate(recs, by="file")
    self.assertEqual(by_file[(100, 0, "transport/p2p.cc")].requested, 180)

  def test_rounding_estimate(self):
    recs = m.parse_log([line("Cuda Calloc", 4096, "0x1"), line("Cuda Calloc", 2 * MiB, "0x2"),
                        line("Cuda Calloc", 2 * MiB + 1, "0x3")])
    t, = m.aggregate(recs, granularity=2 * MiB).values()
    self.assertEqual(t.requested, 4096 + 4 * MiB + 1)
    self.assertEqual(t.rounded_estimate, 2 * MiB + 2 * MiB + 4 * MiB)
    self.assertEqual(t.rounding_estimate, 8 * MiB - (4096 + 4 * MiB + 1))

  def test_no_granularity_means_no_rounding(self):
    t, = m.aggregate(m.parse_log([line("Cuda Calloc", 4096, "0x1")])).values()
    self.assertEqual(t.rounding_estimate, 0)


def snap(stage, free, stats, pid=100, dev=0):
  return {"pid": pid, "dev": dev, "stage": stage, "cuda_free": free, "cuda_total": 80 << 30, "nccl_stats": stats}


class ReconcileTest(unittest.TestCase):
  def setUp(self):
    self.recs = m.parse_log([marker("init"), line("Cuda Calloc", 4096, "0x1"), line("Cuda Calloc", 1 * MiB, "0x2"),
                             marker("allreduce"), line("Cuda Calloc", 3 * MiB, "0x3")])

  def test_gaps_signed_not_assumed_zero(self):
    snaps = [snap("before_init", 1000 * MiB, {"total": 0}),
             snap("init", 990 * MiB, {"total": 4 * MiB}),
             snap("allreduce", 980 * MiB, {"total": 8 * MiB})]
    rows = m.reconcile(self.recs, snaps, granularity=2 * MiB)
    init, ar = rows[1], rows[2]
    self.assertEqual(init["observed_requested"], 4096 + MiB)
    self.assertEqual(init["observed_rounded_estimate"], 4 * MiB)
    self.assertEqual(init["tracked_minus_observed"], 4 * MiB - (4096 + MiB))
    self.assertEqual(init["tracked_minus_rounded_estimate"], 0)
    self.assertEqual(init["device_used_delta"], 10 * MiB)
    self.assertEqual(init["device_minus_tracked"], 6 * MiB)
    # Allreduce: tracked (8 MiB) below the rounding estimate (8 MiB) is 0; observed is cumulative.
    self.assertEqual(ar["observed_requested"], 4096 + 4 * MiB)
    self.assertEqual(ar["tracked_minus_rounded_estimate"], 0)

  def test_negative_gaps_reported_not_clamped(self):
    snaps = [snap("before_init", 1000 * MiB, {"total": 0}),
             snap("init", 1001 * MiB, {"total": 1024})]  # free memory went up; tracked below observed
    init = m.reconcile(self.recs, snaps)[1]
    self.assertEqual(init["device_used_delta"], -MiB)
    self.assertLess(init["tracked_minus_observed"], 0)
    self.assertLess(init["device_minus_tracked"], 0)

  def test_stats_unavailable_is_not_zero(self):
    snaps = [snap("before_init", 1000 * MiB, "unavailable"), snap("init", 990 * MiB, "unavailable")]
    init = m.reconcile(self.recs, snaps)[1]
    self.assertIsNone(init["tracked_total"])
    self.assertIsNone(init["tracked_minus_observed"])
    self.assertIsNone(init["device_minus_tracked"])
    self.assertEqual(init["device_used_delta"], 10 * MiB)

  def test_without_stats_section(self):
    snaps = [snap("init", 990 * MiB, None)]
    self.assertIsNone(m.reconcile(self.recs, snaps)[0]["tracked_total"])

  def test_multiple_processes_and_devices_kept_apart(self):
    recs = m.parse_log([marker("init", 100), line("Cuda Calloc", 10, "0x1", pid=100, dev=0),
                        line("Cuda Calloc", 20, "0x1", pid=100, dev=1),
                        marker("init", 200), line("Cuda Calloc", 40, "0x1", pid=200, dev=0)])
    snaps = [snap("init", 0, {"total": 0}, pid=100, dev=0), snap("init", 0, {"total": 0}, pid=100, dev=1),
             snap("init", 0, {"total": 0}, pid=200, dev=0)]
    self.assertEqual([r["observed_requested"] for r in m.reconcile(recs, snaps)], [10, 20, 40])


def shareable(size, ptr, pid=100, dev=0, tid=105):
  # INFO_LOC(NCCL_P2P | NCCL_ALLOC, "Allocated shareable buffer %p size %zu ipcDesc %p", ...) in src/transport/p2p.cc
  return ("node1:%d:%d [%d] NCCL INFO transport/p2p.cc:258 (ncclP2pAllocateShareableBuffer) "
          "Allocated shareable buffer %s size %d ipcDesc 0x7f00c400b430\n" % (pid, tid, dev, ptr, size))


class ShareableBufferTest(unittest.TestCase):
  def test_cumem_path_counted(self):
    r, = m.parse_log([shareable(512 * MiB, "0xb62000000")])
    self.assertTrue(r.is_gpu and r.ok)
    self.assertEqual((r.kind, r.size, r.ptr), (m.SHAREABLE_KIND, 512 * MiB, "0xb62000000"))
    self.assertEqual(r.site, "transport/p2p.cc:258 (ncclP2pAllocateShareableBuffer)")

  def test_cudamalloc_path_not_double_counted(self):
    # Without cuMem, ncclCudaCalloc logs the buffer first, then the shareable-buffer line repeats its pointer.
    recs = m.parse_log([line("Cuda Calloc", 4 * MiB, "0xa00000", file="transport/p2p.cc", lineno=249,
                             func="ncclP2pAllocateShareableBuffer"),
                        shareable(4 * MiB, "0xa00000")])
    self.assertEqual([r.ok for r in recs], [True, False])
    self.assertFalse(any(r.possible_reuse for r in recs))
    self.assertEqual(sum(t.requested for t in m.aggregate(recs).values()), 4 * MiB)

  def test_same_pointer_later_is_possible_reuse_not_duplicate(self):
    recs = m.parse_log([shareable(MiB, "0xa00000"), line("Cuda Calloc", 8, "0xb00000"), shareable(MiB, "0xa00000")])
    self.assertEqual([r.ok for r in recs], [True, True, True])
    self.assertEqual([r.possible_reuse for r in recs], [False, False, True])


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def real(name):
  with open(os.path.join(DATA, name)) as f:
    return m.parse_log(f)


class RealLogTest(unittest.TestCase):
  """Excerpts of real NCCL v2.32.3-1 logs from a Kaggle T4 run (see the header of each file)."""

  def test_single_gpu_excerpt(self):
    recs = real("kaggle_t4_ng1_rc0.log")
    gpu = [r for r in recs if r.is_gpu]
    self.assertEqual(len(gpu), 7)
    self.assertEqual({r.kind for r in gpu}, {"Cuda CallocAsync", m.SHAREABLE_KIND})
    self.assertTrue(all(r.ok and not r.possible_reuse for r in gpu))
    self.assertEqual({r.memtype for r in gpu if r.kind == "Cuda CallocAsync"}, {2})
    self.assertEqual({r.stage for r in gpu}, {"init"})
    self.assertEqual([r.size for r in gpu if r.kind == m.SHAREABLE_KIND], [512 * MiB])
    # The proxy thread logs the shareable buffer: different tid, same pid and device.
    self.assertEqual(len({(r.pid, r.dev) for r in gpu}), 1)
    self.assertEqual(len({r.tid for r in gpu}), 2)

  def test_single_gpu_tracked_total_excludes_shareable_buffer(self):
    # In this run ncclCommMemStats stays at 388 MiB while the 512 MiB shareable buffer is logged, so tracked
    # minus observed goes negative. The tool must report that, not hide it.
    with open(os.path.join(DATA, "kaggle_t4_ng1_rc0_stats.json")) as f:
      snaps = json.load(f)["snapshots"]
    init = [r for r in m.reconcile(real("kaggle_t4_ng1_rc0.log"), snaps, 2 * MiB) if r["stage"] == "init"][0]
    self.assertEqual(init["tracked_total"], 388 * MiB)
    self.assertLess(init["tracked_minus_observed"], 0)
    self.assertEqual(init["device_used_delta"], 970 * MiB)

  def test_two_gpu_shm_run(self):
    recs = real("kaggle_t4_ng2_rc1.log")
    self.assertEqual(len(recs), 16)
    self.assertEqual({r.dev for r in recs}, {0, 1})
    self.assertTrue(all(r.is_gpu and r.ok and r.stage == "init" for r in recs))
    per_dev = collections.Counter()
    for (pid, dev, _), t in m.aggregate(recs, 2 * MiB).items():
      per_dev[dev] += t.rounded_estimate
    # Matches ncclStatGpuMemTotal (16 MiB per device) reported in that run.
    self.assertEqual(per_dev, {0: 16 * MiB, 1: 16 * MiB})


class CliTest(unittest.TestCase):
  def test_end_to_end(self):
    with tempfile.TemporaryDirectory() as d:
      log = os.path.join(d, "nccl.log")
      stats = os.path.join(d, "stats.json")
      with open(log, "w") as f:
        f.writelines([marker("init"), line("Cuda Calloc", 4096, "0x1"), line("Ib Alloc", 4096, "0x9"),
                      line("Cuda Alloc", 8, "(nil)")])
      with open(stats, "w") as f:
        json.dump({"snapshots": [snap("before_init", 100 * MiB, {"total": 0}),
                                 snap("init", 96 * MiB, {"total": 2 * MiB})]}, f)
      out = io.StringIO()
      with redirect_stdout(out):
        self.assertEqual(m.main([log, "--stats", stats, "--granularity", "2MiB"]), 0)
      text = out.getvalue()
      self.assertIn("transport/p2p.cc:500 (p2pSendSetup)", text)
      self.assertIn("2 GPU allocation lines, 1 failed or zero-size (not counted), 1 host allocation lines", text)
      self.assertIn("| 100 | 0 | init |", text)

  def test_parse_size(self):
    self.assertEqual(m.parse_size("2MiB"), 2 * MiB)
    self.assertEqual(m.parse_size("64k"), 64 << 10)
    self.assertEqual(m.parse_size("4096"), 4096)
    with self.assertRaises(ValueError):
      m.parse_size("two")


if __name__ == "__main__":
  unittest.main()
