#!/usr/bin/env bash
# Reports two DIFFERENT fragmentation signals for zone Normal. Run remotely
# by prelaunch_flush.sh. Output: line 1 = nr_free_pages_blocks aggregate
# (from /proc/zoneinfo), line 2 = "order4_count order5_count" (from
# /proc/buddyinfo).
#
# Why the aggregate (line 1) alone is not enough: found 2026-09-21
# (recipes/VALIDATION.md, "Root cause found: head-node-only medium-order
# fragmentation, not a memory leak"). The aggregate sums free blocks across
# EVERY order and is dominated by abundant small-order (4-32 KB) blocks --
# it can read in the hundreds of thousands to tens of millions even while
# medium orders (order-4/64KB, order-5/128KB specifically) are individually
# down to ~1 block each. That partial, medium-order-specific collapse is
# exactly what caused 3 real earlyoom kills on the head node this project
# hit in a single ~30 hour window, and this check reported "OK" (via the
# aggregate alone) before every single one of those boots. Line 2 exists to
# close that gap -- see prelaunch_flush.sh's check_fragmentation for the
# threshold this is checked against.
#
# Why the aggregate (line 1) is still worth keeping, not just replacing:
# the 2026-09-08 host/driver-level trace of the 240K NVRM
# `_memdescAllocInternal` NV_ERR_NO_MEMORY failure found that raw
# MemAvailable% looked similar (~2-4.5%) in both a successful and a failed
# boot -- the field that discriminated THAT incident was this aggregate
# dropping to exactly 0 (a much more total exhaustion than the medium-order
# case line 2 targets). See the host_fragmentation_xid31_reboot_risk
# memory note for that original incident.
set -euo pipefail

awk '
  /^Node/ && /zone/ { in_normal = ($NF == "Normal") }
  in_normal && /nr_free_pages_blocks/ { print $2; exit }
' /proc/zoneinfo

awk '$4 == "Normal" { print $9, $10; exit }' /proc/buddyinfo
